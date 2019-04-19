from __future__ import unicode_literals
import sys, os, threading, json, platform

from readings import *
from anemUIProcessor import *
import anemUIProcessor as ai
from  anemUIWindow import *
import anemUIWindow as aw
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import readings as readings

v = [0, 0, 0]
config_num_sensors = [0]
config_keep_raw_data_button=[0]
example_size=[0]
#current_time = str(datetime.now().strftime("%Y-%m-%d"))
current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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
        print("---------- Raw Data Length -------- ", len(alldata))    

        if config_keep_raw_data_button[0]=='n':
            n=len(alldata)
            if n>100:
                del alldata[0:n-2]
                print("---------- Clean Raw Data ! -------- ")  
                
#        alldata=alldata_pool


# Generator function that runs the data streaming script and parses the incoming data, yielding the DecodedRawInput objects. Optionally filters for certain IDs and sites.
def read_input(anemometer_ids=None, site_filter=None, usb_port=None):
    operating_system = platform.system()
    print("OS: ", operating_system)
    if operating_system is "Windows":
        open_args = [resource_path("./input/src.exe")]
    else:
        open_args = [resource_path("./input/src")]
    if usb_port is not None:
        open_args.append(usb_port)
        
#    open_args = ["python", "input.py"]  # if you want to simulate input from test dataset

    if site_filter is None:
        p = Popen(open_args, stdout=PIPE)
    else:
        new_env = os.environ.copy() # Is os.environ.copy() necessary?
        new_env['SITE_FILTER'] = site_filter
        p = Popen(open_args, env=new_env, stdout=PIPE)
    
    data_prefix = "MIRROR_STDOUT"
#    num_sensors = 4
#    num_sensors = 6 # set sensor number is 6 for updated duct

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
#        print('+++++++++++++++++++++', site)
        is_duct = bool(data['RawInput']['SetInfo']['IsDuct'])
        if 'IsDuct6' in data['RawInput']['SetInfo']:
            is_duct6 = bool(data['RawInput']['SetInfo']['IsDuct6'])
            is_room = bool(data['RawInput']['SetInfo']['IsRoom'])
        else: # earlier versions of go algorithm only have 'IsDuct'. 
            is_duct6 = False
            is_room = not is_duct
        if is_duct6 == True:
            config_num_sensors[0] = 6 
        else:
            config_num_sensors[0] = 4
                  
            
        
        parsed = DecodedRawInput(anemometer_id, timestamp, site, is_duct, is_duct6, is_room, v)
        success = True
        temps = []
        accelerometers = []
        magnetometers = []
        for sensor in range(0, config_num_sensors[0]):
            sensor_data = data['RawInput']['ChirpHeaders'][sensor]
            if sensor_data is None:
                print("Data is incomplete, missing sensor ", sensor, ". Skipping. Data line received:\n", line_orig)
                success = False
                break
            parsed_sensor_data = DecodedChirpHeader(sensor, sensor_data['MaxIndex'],
                                                    sensor_data['QValues'], sensor_data['IValues'])
            temps.append(float(sensor_data['Temperature']))
            accels = sensor_data['Accelerometer']
            mags = sensor_data['Magnetometer']
            accelerometers.append([float(x) for x in accels])
            magnetometers.append([float(x) for x in mags])
            # print(parsed_sensor_data.id, parsed_sensor_data.max_indices, parsed_sensor_data.real, parsed_sensor_data.imaginary)
            parsed.add_chirp_header(parsed_sensor_data)
        if success:
            parsed.temperature = temps[0]  # all temp, accelerometer, and magnetometer fields per sensor are equal.
            parsed.accelerometer = accelerometers[0]
            parsed.magnetometer = magnetometers[0]
            yield (parsed, line_orig)

    # When the subprocess terminates there might be unconsumed output that still needs to be processed.
    print(p.stdout.read())  # (? probably unnecessary)


def read_config():
    id_file = open(resource_path("config.txt"), "r")
    duct_anemometer_ids = []
    room_anemometer_ids = []
    min = True
    duct_dist = None
    site_id = None
    usb_port = None
    use_saved_calibration = False 
    for line in id_file:
        words = line.split()
        if len(words) == 3:
            if words[0] == "duct" and (words[1] == "distance" or words[1] == "distance:"):
                try:
                    duct_dist = float(words[2])
                except ValueError:
                    print("Error: couldn't parse given duct distance of " + words[2] + ". Using default.")
        elif len(words) == 2:
            if words[0].lower()=="numsensor" or words[0].lower()=="numsensor:":
                config_num_sensors[0]=int(words[1])
#                print("----sensor number-------",config_num_sensors)
            if words[0].lower()=="keep_raw_data?" or words[0].lower()=="keep_raw_data?:":
                config_keep_raw_data_button[0] = words[1].lower()             
#                print("----config_keep_raw_data_button[0]-------",config_keep_raw_data_button[0])   
            if words[0].lower()=="dump_process_data" or words[0].lower()=="edump_process_data:":
                example_size[0]=int(words[1])
            if words[0] == "site:":
                site_id = words[1]    # Assumes siteID has no spaces
                if site_id == "None":
                    usb_port = None
            elif words[0] == "USBport:":
                usb_port = words[1]
                if usb_port == "None":
                    usb_port = None
            elif words[0] == "room" or words[0] == "room:":
                if words[1].lower() == "min":
                    min = True
                elif words[1].lower() == "max":
                    min = False                    
                else:
                    print("Error: unrecognized room filter scheme; should be 'room min' or 'room max'")
            elif words[1] == "duct":
                duct_anemometer_ids.append(words[0])
            elif words[1] == "room":
                room_anemometer_ids.append(words[0])
        elif len(words) == 4:
            if words[0].lower() == "use" and words[1].lower() == "saved" and (words[2] == "calibration" or words[2] == "calibration:"):
                use_saved_calibration = words[3].lower() == "true"
            if words[0].lower()=="mag" or words[0].lower()=="mag:":
                v[0]=int(words[1])
                v[1]=int(words[2])
                v[2]=int(words[3])


    return duct_anemometer_ids, room_anemometer_ids, site_id, usb_port, min, duct_dist, use_saved_calibration


def dump_raw_data():


    print("Dumping all data ")
    f = open(resource_path('anemometer_raw_data.txt'), 'w')
    f.write(current_time+'\n')
    for line in alldata:
        f.write(line)
        f.write('\n')
    f.close()
    print("Successfully dumped all data to anemometer_raw_data.txt")


def _create_processors_and_windows():
    calibration_period_room = 10
    calibration_period_duct = 30
    for id in duct_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, True, dump_raw_data, calibration_period_duct, INCLUDE_CALIBRATION, use_room_min, duct_dist, use_saved_calibration)
        mainWindow.add_window(anemometer_processors[id].generate_window())
    for id in room_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, False, dump_raw_data, calibration_period_room, INCLUDE_CALIBRATION, use_room_min, duct_dist, use_saved_calibration)
        mainWindow.add_window(anemometer_processors[id].generate_window())

if __name__ == '__main__':
    
    alldata = []  # list of all input data lines as is
    anemometer_processors = {}  # {anemometer_id : AnemometerProcessor}
    all_anemometer_ids = []
    INCLUDE_CALIBRATION = False

    # Set up app
    read_config()
    readings.v=v
    print('---- config_num_sensors ----',config_num_sensors)
    ai.num_sensors=config_num_sensors[0]
    aw.num_sensors=config_num_sensors[0]
    print("----config_keep_raw_data_button[0]-------",config_keep_raw_data_button[0])            
    aw.config_keep_rawdata_button=config_keep_raw_data_button[0]
    aw.example_size=example_size[0]
    print("-----example_size-----", example_size)
    qApp = QApplication(sys.argv)
    mainWindow = MultipleApplicationWindows()         


    # Set up anemometer stream processors
    duct_anemometer_ids, room_anemometer_ids, site_id, usb_port, use_room_min, duct_dist, use_saved_calibration = read_config()
    all_anemometer_ids = duct_anemometer_ids + room_anemometer_ids
    print('Starting anemometers...............................................' )    
    print("Tracking duct anemometers: ", duct_anemometer_ids)
    print("Tracking room anemometers: ", room_anemometer_ids)
#    _create_processors_and_windows()
  
    # Start thread for reading in input
    t = threading.Thread(target=_process_input)
    t.daemon = True
    t.start()
    _create_processors_and_windows()
#     
    # Start the app
    
    mainWindow.setWindowTitle("Main hub")
    splash = WM()
    mainWindow.show()
    splash.show()

    sys.exit(qApp.exec_())
