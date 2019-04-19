from datetime import datetime
import numpy as np
import anemUI as anemUI
import anemUIWindow as aw
v = []
squelch = False  # True if you want to squelch warning messages. (TODO: There's probably a nicer way to do this.)

class DecodedChirpHeader:
    def __init__(self, id, max_indices, real, imaginary):
        self.id = id
        self.max_indices = max_indices  # list
        self.real = real  # list of lists
        self.imaginary = imaginary  # list of lists


class DecodedRawInput:
    def __init__(self,  anemometer_id="", timestamp=0, site="", is_duct=False, is_duct6=False, is_room=False,
                 temperature=0, accelerometer=None, magnetometer=None, num_sensors=6, chirp_headers=None):
        if chirp_headers is None:
            chirp_headers = []
        self.anemometer_id = anemometer_id
        self.timestamp = timestamp
        self.site = site
        self.is_duct = is_duct
        self.is_duct6 = is_duct6
        self.is_room = is_room
        
        if is_duct6== True:
            self.num_sensors = 6
            anemUI.num_sensors =6
            aw.num_sensors=6
            is_duct == True
        else:
            self.num_sensors = 4
            anemUI.num_sensors =4
            aw.num_sensors=4
            
            
            
        self.chirp_headers = chirp_headers
        self.temperature = temperature
        self.accelerometer = accelerometer
        self.magnetometer = magnetometer

    def add_chirp_header(self, chirp_header):
        self.chirp_headers.append(chirp_header)

    def get_max_index(self, src, dst):
        return self.chirp_headers[src].max_indices[dst]

    def get_real(self, src, dst):
        return self.chirp_headers[src].real[dst]

    def get_imaginary(self, src, dst):
        return self.chirp_headers[src].imaginary[dst]

    def get_temperature(self):
        return self.temperature

    def get_accelerometer(self):
        return self.accelerometer

    def get_magnetometer(self):
        return self.magnetometer
    
    def get_num_sensor(self):
        return self.num_sensors
    

    # Uses accelerometer and magnetometer values to determine roll, pitch, and yaw of anemometer in radians
    # see paper: https://www.nxp.com/files-static/sensors/doc/app_note/AN4248.pdf
    def get_rotations(self):
        # Operations in order of yaw, then pitch, then roll
        # V, the Hard-Iron vector, has placeholder values for now.
#        v = [-10,+110, -0] # -50,+60
#        v = [-10,110,0] # -50,+60
#        print("Hard-iron coeffients from config.txt", v)
        
#        yaw_bias =[0,90,160,180,-138]
#        soft_yaw =[[1.2857,-25.714],[1,-340],[0.678,-3.2203],[1.667,140],[0.8738,8.7379]]
        
        g = self.accelerometer
        b = self.magnetometer
        aw.mag=b
        
        roll = np.arcsin(g[0]/np.sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]))   # eqn 13
        pitch = np.arcsin(-g[1]/np.sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]))
        temp_yaw = np.arctan2((b[2] - v[2]) * np.sin(roll) - (b[0] - v[0]) * np.cos(roll),
                         (b[1] - v[1]) * np.cos(pitch) + (b[0] - v[0])* np.sin(roll) * np.sin(pitch) +
                         (b[2] - v[2]) * np.cos(roll) * np.sin(pitch))

#        pitch = 180 * np.arctan2(g[0], np.sqrt(g[1]*g[1] + g[2]*g[2]))/np.pi;
#        roll =  180 * np.arctan2(g[1], np.sqrt(g[0]*g[0] + g[2]*g[2]))/np.pi;
#
#        mag_x = b[0]*np.cos(pitch) + b[1]*np.sin(roll)*np.sin(pitch) + b[2]*np.cos(roll)*np.sin(pitch)
#        mag_y = b[1] * np.cos(roll) - b[2] * np.sin(roll)
#        yaw = 180 * np.arctan2(-mag_y,mag_x)/np.pi;

        yaw = temp_yaw        
#        if temp_yaw>=90 and temp_yaw < 160:
#            yaw = temp_yaw * soft_yaw[0][0] + soft_yaw[0][1]           
#        if temp_yaw>=160 and temp_yaw < 180:
#            yaw = temp_yaw * soft_yaw[1][0] + soft_yaw[1][1]            
#        if temp_yaw>=-138 and temp_yaw <-10:
#            yaw = temp_yaw * soft_yaw[2][0] + soft_yaw[2][1]  
#        if temp_yaw>=-180 and temp_yaw <-138:
#            yaw = temp_yaw * soft_yaw[3][0] + soft_yaw[3][1]                        



#        with open("Debug_data.txt", 'a') as f:
#            f.write(str( yaw)+'\n' )   
#            f.flush()
        
#        print('-Line 64 in readings.py--roll, pitch, yaw -----',roll, pitch, yaw)
        

        return roll, pitch, yaw

    # Returns change of coordinates matrix R from world space to anemometer space, s.t. v_anem = R * v_world
    def get_rotation_matrix(self):
        
        roll, pitch, yaw = self.get_rotations()
        aw.show_pitch=pitch
        aw.show_yaw= yaw

#        # v_anem = Rx * Ry * Rz * v_world
        r_x = np.matrix([[1, 0, 0],
                         [0, np.cos(roll), np.sin(roll)],
                         [0, -np.sin(roll), np.cos(roll)]])
        r_y = np.matrix([[np.cos(pitch), 0, -np.sin(pitch)],
                         [0, 1, 0],
                         [np.sin(pitch), 0, np.cos(pitch)]])
        r_z = np.matrix([[np.cos(yaw), np.sin(yaw), 0],
                         [-np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    
        return np.dot(r_x, np.dot(r_y, r_z))

# old implementation
# class PathReading:
#     EXPECTED_DATA_COUNT = 16 # expected 16 readings per real/im/mag for each path
#
#     def __init__(self, path_string, datetime_obj, freq=-1, calres=-1, tof=-1, real=None, im=None, mag=None):
#         if real is None:
#             real = []
#         if im is None:
#             im = []
#         if mag is None:
#             mag = []
#         self.path_string = path_string
#         self.datetime_obj = datetime_obj
#
#         self.freq = freq
#         self.calres = calres
#         self.tof = tof
#         self.real = real
#         self.im = im
#         self.mag = mag
#         self.src, self.dst = PathReading.verify_path(path_string)
#         if self.src == -1 or self.dst == -1:
#             print("Error: path string not formatted as expected: ", self.path_string)
#         if self.src == self.dst and self.tof != -1:
#             print("Warn: did not expect tof for self path: ", self.tof)
#
#     def set_freq(self, val):
#         if self.freq != -1:
#             if not squelch:
#                 print("Warn: frequency already set. Ignored")
#             return
#         self.freq = val
#
#     def set_calres(self, val):
#         if self.calres != -1:
#             if not squelch:
#                 print("Warn: calres already set. Ignored")
#             return
#         self.calres = val
#
#     def set_tof(self, val):
#         if self.tof != -1:
#             if not squelch:
#                 print("Warn: time of flight already set. Ignored")
#             return
#         self.tof = val
#
#     def add_real(self, val):
#         if len(self.real) >= PathReading.EXPECTED_DATA_COUNT:
#             if not squelch:
#                 print("Warn: Attempted to add more real readings than expected. Ignored", self.path_string)
#             return
#         self.real.append(val)
#
#     def add_im(self, val):
#         if len(self.im) >= PathReading.EXPECTED_DATA_COUNT:
#             if not squelch:
#                 print("Warn: Attempted to add more im readings than expected. Ignored", self.path_string)
#             return
#         self.im.append(val)
#
#     def add_mag(self, val):
#         if len(self.mag) >= PathReading.EXPECTED_DATA_COUNT:
#             if not squelch:
#                 print("Warn: Attempted to add more mag readings than expected. Ignored", self.path_string)
#             return
#         self.mag.append(val)
#
#     @staticmethod
#     # Returns path's src and dst, if valid format src_to_dst
#     def verify_path(str):
#         i = str.find('_to_')
#         if i == -1:
#             return (-1, -1)
#         return (int(str[0:i]), int(str[i+4:]))
#
#     def is_complete(self):
#         return len(self.real) >= PathReading.EXPECTED_DATA_COUNT and len(self.im) >= PathReading.EXPECTED_DATA_COUNT and len(self.mag) >= PathReading.EXPECTED_DATA_COUNT
#
#
# class Reading:
#     EXPECTED_PATH_COUNT = 16  # [0-3] to [0-3], so 16 paths
#
#     def __init__(self, anemometer_id, path_readings=None):
#         if path_readings is None:
#             path_readings = {}
#         self.anemometer_id = anemometer_id
#         self.path_readings = path_readings
#
#     def path_initialized(self, path_string):
#         if path_string not in self.path_readings:
#             print("Error: Path reading not yet initialized, discarding reading")
#             return False
#         return True
#
#     def is_complete(self):
#         if len(self.path_readings) < Reading.EXPECTED_PATH_COUNT:
#             return False
#         for _, path_reading in self.path_readings.items():
#             if not path_reading.is_complete():
#                 return False
#         return True
#         # return "3_to_3" in self.path_readings and self.path_readings["3_to_3"].is_complete()
#
#     # accessors to add readings
#     def add_path_reading(self, path_string, datetime_obj):
#         if path_string in self.path_readings:
#             print("Warn: path already tracked in this reading; overwriting!")
#         if len(self.path_readings) >= Reading.EXPECTED_PATH_COUNT:
#             print("Warn: Adding more paths than expected. Continuing anyways")
#         self.path_readings[path_string] = PathReading(path_string, datetime_obj)
#
#     def set_freq(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].set_freq(val)
#
#     def set_calres(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].set_calres(val)
#
#     def set_tof(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].set_tof(val)
#
#     def add_real(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].add_real(val)
#
#     def add_im(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].add_im(val)
#
#     def add_mag(self, path_string, val):
#         if not self.path_initialized(path_string):
#             return
#         self.path_readings[path_string].add_mag(val)
#
