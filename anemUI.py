from __future__ import unicode_literals
import sys, os, threading, json, platform
from PyQt5 import QtCore, QtWidgets
from readings import *
from anemUIProcessor import *
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT
import numpy as np


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def _process_input():
    readings_generator = read_input(all_anemometer_ids, site_id, usb_port)
    for (reading, line) in readings_generator:
        # if len(line) > 0:
        #     print("read ", line)
        alldata.append(line)
        anemometer_processors[reading.anemometer_id].process_reading(reading)


# Generator function that runs the data streaming script and parses the incoming data, yielding the DecodedRawInput objects. Optionally filters for certain IDs and sites.
def read_input(anemometer_ids=None, site_filter=None, usb_port=None):
    operating_system = platform.system()
    print("OS: ", operating_system)
    if operating_system is "Windows":
        open_args = ["./input/src.exe"]
    else:
        open_args = ["./input/src"]
    if usb_port is not None:
        open_args.append(usb_port)
    open_args = ["python", "input.py"] # if you want to simulate input from test dataset

    if site_filter is None:
        p = Popen(open_args, stdout=PIPE)
    else:
        new_env = os.environ.copy() # Is os.environ.copy() necessary?
        new_env['SITE_FILTER'] = site_filter
        p = Popen(open_args, env=new_env, stdout=PIPE)
    
    data_prefix = "MIRROR_STDOUT"
    num_sensors = 4

    while True:
        line_orig = p.stdout.readline().decode("utf-8")
        line = line_orig[0:-1] # strip trailing newline

        # not a data line, skip
        if len(line) < len(data_prefix) or line[0:len(data_prefix)] != data_prefix:
            continue

        line = line[len(data_prefix) + 1:]  # trim off leading label
        data = json.loads(line)
        anemometer_id = data['Sensor']
        # If not an anemometer of interest, skip
        if anemometer_ids is not None and anemometer_id not in anemometer_ids:
            continue

        timestamp = data['Timestamp']/1e9  # timestamp to seconds
        site = data['RawInput']['SetInfo']['Site']
        is_duct = bool(data['RawInput']['SetInfo']['IsDuct'])
        if 'IsDuct6' in data['RawInput']['SetInfo']:
            is_duct6 = bool(data['RawInput']['SetInfo']['IsDuct6'])
            is_room = bool(data['RawInput']['SetInfo']['IsRoom'])
        else: # earlier versions of go algorithm only have 'IsDuct'. 
            is_duct6 = False
            is_room = not is_duct
        
        parsed = DecodedRawInput(anemometer_id, timestamp, site, is_duct, is_duct6, is_room)
        success = True
        temps = []
        for sensor in range(0, num_sensors):
            sensor_data = data['RawInput']['ChirpHeaders'][sensor]
            if sensor_data is None:
                print("Data is incomplete, missing sensor ", sensor, ". Skipping. Data line received:\n", line_orig)
                success = False
                break
            temps.append(float(sensor_data['Temperature']))
            parsed_sensor_data = DecodedChirpHeader(sensor, sensor_data['MaxIndex'],
                                                    sensor_data['QValues'], sensor_data['IValues'])
            # print(parsed_sensor_data.id, parsed_sensor_data.max_indices, parsed_sensor_data.real, parsed_sensor_data.imaginary)
            parsed.add_chirp_header(parsed_sensor_data)
        if success:
            parsed.temperature = temps[0] # all temperature fields are equal.
            yield (parsed, line_orig)

    # When the subprocess terminates there might be unconsumed outputthat still needs to be processed.
    print(p.stdout.read())  # (? probably unnecessary)

def read_anemometer_IDs():
    id_file = open("anemometerIDs.txt", "r")
    duct_anemometer_ids = []
    room_anemometer_ids = []
    site_id = ""
    usb_port = ""
    for line in id_file:
        words = line.split()
        if len(words) == 2:
            if words[0] == "site:":
                site_id = words[1]    # Assumes siteID has no spaces
                if site_id == "None":
                    usb_port = None
            elif words[0] == "USBport:":
                usb_port = words[1]
                if usb_port == "None":
                    usb_port = None
            elif words[1] == "duct":
                duct_anemometer_ids.append(words[0])
            elif words[1] == "room":
                room_anemometer_ids.append(words[0])

    return (duct_anemometer_ids, room_anemometer_ids, site_id, usb_port)
    


def dump_raw_data():
    print("Dumping all data")
    f = open(resource_path("anemometer_raw_data.txt"), 'w')
    for line in alldata:
        f.write(line)
        f.write('\n')
    f.close()
    print("Successfully dumped all data to anemometer_raw_data.txt")



def _create_processors_and_windows():
    for id in duct_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, True, dump_raw_data, CALIBRATION_PERIOD, INCLUDE_CALIBRATION)
        mainWindow.add_window(anemometer_processors[id].generate_window())
    for id in room_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, False, dump_raw_data, CALIBRATION_PERIOD, INCLUDE_CALIBRATION)
        mainWindow.add_window(anemometer_processors[id].generate_window())


if __name__ == '__main__':
    alldata = []  # list of all input data lines as is
    anemometer_processors = {}  # {anemometer_id : AnemometerProcessor}
    site_id = None
    usb_port = None
    duct_anemometer_ids = []
    room_anemometer_ids = []
    all_anemometer_ids = []
    CALIBRATION_PERIOD = 10  # number of readings to consider calibration period
    INCLUDE_CALIBRATION = False

    # Set up app
    qApp = QtWidgets.QApplication(sys.argv)
    mainWindow = MultipleApplicationWindows()
    mainWindow.setWindowTitle("Main hub")

    # Set up anemometer stream processors
    duct_anemometer_ids, room_anemometer_ids, site_id, usb_port = read_anemometer_IDs()
    all_anemometer_ids = duct_anemometer_ids + room_anemometer_ids
    print("Tracking duct anemometers: ", duct_anemometer_ids)
    print("Tracking room anemometers: ", room_anemometer_ids)
    _create_processors_and_windows()

    # Start thread for reading in input
    t = threading.Thread(target=_process_input)
    t.daemon = True
    t.start()

    # Start the app
    mainWindow.show()
    sys.exit(qApp.exec_())
