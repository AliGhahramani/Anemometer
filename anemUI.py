from __future__ import unicode_literals
import sys
import os
from PyQt5 import QtCore, QtWidgets
import threading
import json
from readings import *
from anemUIProcessor import *
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

mainWindow = None
alldata = []  # list of all input data lines as is
anemometer_processors = {}  # {anemometer_id : AnemometerProcessor}
CALIBRATION_PERIOD = 10  # number of readings to consider calibration period
INCLUDE_CALIBRATION = False

# is_duct = True
duct_anemometer_ids = []
room_anemometer_ids = []
all_anemometer_ids = []
executable_path = "./anemo"
# executable_path = "./anemomteer-master"
# executable_path = "./src"

# OLD VERSION
# def read_input():
#     # p = Popen([executable_path], stdout=PIPE)
#     p = Popen(["python", "input.py"], stdout=PIPE)  # if you want to simulate input from test dataset
#
#     while True:
#         line = p.stdout.readline().decode("utf-8")[0:-1]  # strip newline at end
#         alldata.append(line)
#         words = line.split()
#
#         # If line doesn't look like data, or not anemometer we're tracking, pass
#         if len(words) != 6 or (words[4].count('/') < 3):
#             continue
#         anemometer_id = words[4].split('/')[2]
#         if anemometer_id not in all_anemometer_ids:
#             continue
#         path_string = words[4][-6:]
#         # tof data is formatted differently than other data
#         if "tof" in words[4]:
#             path_string = words[5][0:words[5].find("=")]
#         src, dst = PathReading.verify_path(path_string)
#         # If couldn't find path string, pass
#         if src == -1 or dst == -1:
#             continue
#         # TODO: There's a lot of rehashing here, see if python allows to keep a reference instead
#         # hack -- sometimes readings drop certain paths. If paths are dropped, drop the reading.
#         if path_string == "0_to_0" and len(readings[anemometer_id].path_readings) > 1:
#             readings[anemometer_id] = Reading(anemometer_id)
#         if path_string == "0_to_1" and len(readings[anemometer_id].path_readings) > 2:
#             readings[anemometer_id] = Reading(anemometer_id)
#
#         # if first occurence of this path in this reading, track datetime
#         if path_string not in readings[anemometer_id].path_readings:
#             datetime_str = words[0] + ' ' + words[1][0:-3] + ' ' + words[2] + ' ' + words[3]
#             datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f %z %Z')
#             readings[anemometer_id].add_path_reading(path_string, datetime_obj)
#
#         # Save data
#         i = words[5].find('=')
#         if i == -1:
#             print("WARN: Unexpected value format: ", words[5], ". Discarding")
#             continue
#         val_type = words[5][0:i]
#         val = float(words[5][i + 1:])
#         if "raw" in words[4]:
#             if val_type == "real":
#                 readings[anemometer_id].add_real(path_string, val)
#             elif val_type == "im":
#                 readings[anemometer_id].add_im(path_string, val)
#             elif val_type == "mag":
#                 readings[anemometer_id].add_mag(path_string, val)
#             else:
#                 print("WARN: unrecognized raw value type: ", val_type, ". Discarding")
#         elif "freq" in words[4]:
#             readings[anemometer_id].set_freq(path_string, val)
#         elif "calres" in words[4]:
#             readings[anemometer_id].set_calres(path_string, val)
#         elif "tof" in words[4]:
#             readings[anemometer_id].set_tof(path_string, val)  # tof in different format
#         else:
#             print("WARN: unrecognized value type: ", words[4], " ", words[5], ". Discarding")
#
#         if readings[anemometer_id].is_complete():
#             anemometer_processors[anemometer_id].process_reading(readings[anemometer_id])
#             readings[anemometer_id] = Reading(anemometer_id)
#
#     # When the subprocess terminates there might be unconsumed outputthat still needs to be processed.
#     print(p.stdout.read())  # (? probably unnecessary)

def read_input():
    p = Popen([executable_path], stdout=PIPE)
    # p = Popen(["python", "input.py"], stdout=PIPE)  # if you want to simulate input from test dataset
    data_name = "MIRROR_STDOUT"
    num_sensors = 4

    while True:
        line = p.stdout.readline().decode("utf-8")[0:-1]  # strip newline at end
        if len(line) > 0:
            print("read ", line)
        alldata.append(line)

        # not a data line, skip
        if len(line) < len(data_name) or line[0:len(data_name)] != data_name:
            continue

        line = line[len(data_name) + 1:]  # trim off leading label
        data = json.loads(line)
        anemometer_id = data['Sensor']

        # not an anemometer of interest, skip
        if anemometer_id not in all_anemometer_ids:
            continue

        parsed = DecodedRawInput()
        for sensor in range(0, num_sensors):
            sensor_data = data['RawInput']['ChirpHeaders'][sensor]
            parsed_sensor_data = DecodedChirpHeader(sensor, sensor_data['MaxIndex'],
                                                    sensor_data['QValues'], sensor_data['IValues'])
            # print(parsed_sensor_data.id, parsed_sensor_data.max_indices, parsed_sensor_data.real, parsed_sensor_data.imaginary)
            parsed.add_chirp_header(parsed_sensor_data)
        anemometer_processors[anemometer_id].process_reading(parsed)

    # When the subprocess terminates there might be unconsumed outputthat still needs to be processed.
    print(p.stdout.read())  # (? probably unnecessary)

def read_anemometer_IDs():
    global all_anemometer_ids
    id_file = open("anemometerIDs.txt", "r")
    for line in id_file:
        words = line.split()
        if len(words) == 2:
            if words[1] == "duct":
                duct_anemometer_ids.append(words[0])
            elif words[1] == "room":
                room_anemometer_ids.append(words[0])
    all_anemometer_ids = duct_anemometer_ids + room_anemometer_ids
    print("Tracking duct anemometers: ", duct_anemometer_ids)
    print("Tracking room anemometers: ", room_anemometer_ids)


def dump_raw_data():
    print("Dumping all data")
    f = open("./anemometer_raw_data.txt", 'w')
    for line in alldata:
        f.write(line)
        f.write('\n')
    f.close()


def create_processors_readings_windows():
    for id in duct_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, True, dump_raw_data, CALIBRATION_PERIOD, INCLUDE_CALIBRATION)
        mainWindow.add_window(anemometer_processors[id].generate_window())
    for id in room_anemometer_ids:
        anemometer_processors[id] = AnemometerProcessor(id, False, dump_raw_data, CALIBRATION_PERIOD, INCLUDE_CALIBRATION)
        mainWindow.add_window(anemometer_processors[id].generate_window())


# Set up app
qApp = QtWidgets.QApplication(sys.argv)
mainWindow = MultipleApplicationWindows()
mainWindow.setWindowTitle("Main hub")

# Set up anemometer stream processors
read_anemometer_IDs()
create_processors_readings_windows()

# Start thread for reading in input
t = threading.Thread(target=read_input)
t.daemon = True
t.start()

# Start the app
mainWindow.show()
sys.exit(qApp.exec_())
