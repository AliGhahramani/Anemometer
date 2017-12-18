from datetime import datetime

squelch = True # True if you want to squelch warning messages. (TODO: There's probably a nicer way to do this.)

class PathReading:
    EXPECTED_DATA_COUNT = 16 # expected 16 readings per real/im/mag for each path
    def __init__(self, path_string, datetime_obj, 
            freq=-1, calres=-1, tof=-1, real=None, im=None, mag=None):
        if real is None:
            real = []
        if im is None:
            im = []
        if mag is None:
            mag = []
        self.path_string = path_string
        self.datetime_obj = datetime_obj

        self.freq = freq
        self.calres = calres
        self.tof = tof
        self.real = real
        self.im = im
        self.mag = mag
        self.src, self.dst = PathReading.verify_path(path_string)
        if self.src == -1 or self.dst == -1:
            print("Error: path string not formatted as expected: ", self.path_string)
        if self.src == self.dst and self.tof != -1:
            print("Warn: did not expect tof for self path: ", self.tof)

    def set_freq(self, val):
        if self.freq != -1:
            if not squelch:
                print("Warn: frequency already set. Ignored")
            return
        self.freq = val

    def set_calres(self, val):
        if self.calres != -1:
            if not squelch:
                print("Warn: calres already set. Ignored")
            return
        self.calres = val

    def set_tof(self, val):
        if self.tof != -1:
            if not squelch:
                print("Warn: time of flight already set. Ignored")
            return
        self.tof = val

    def add_real(self, val):
        if len(self.real) >= PathReading.EXPECTED_DATA_COUNT:
            if not squelch:
                print("Warn: Attempted to add more real readings than expected. Ignored", self.path_string)
            return
        self.real.append(val)

    def add_im(self, val):
        if len(self.im) >= PathReading.EXPECTED_DATA_COUNT:
            if not squelch:
                print("Warn: Attempted to add more im readings than expected. Ignored", self.path_string)
            return
        self.im.append(val)

    def add_mag(self, val):
        if len(self.mag) >= PathReading.EXPECTED_DATA_COUNT:
            if not squelch:
                print("Warn: Attempted to add more mag readings than expected. Ignored", self.path_string)
            return
        self.mag.append(val)

    @staticmethod
    # Returns path's src and dst, if valid format src_to_dst
    def verify_path(str):
        i = str.find('_to_')
        if i == -1:
            return (-1, -1)
        return (int(str[0:i]), int(str[i+4:]))

    def is_complete(self):
        return len(self.real) >= PathReading.EXPECTED_DATA_COUNT and len(self.im) >= PathReading.EXPECTED_DATA_COUNT and len(self.mag) >= PathReading.EXPECTED_DATA_COUNT

class Reading:
    EXPECTED_PATH_COUNT = 16 # [0-3] to [0-3], so 16 paths
    def __init__(self, anemometer_id, path_readings=None):
        if path_readings is None:
            path_readings = {}
        self.anemometer_id = anemometer_id
        self.path_readings = path_readings

    def path_initialized(self, path_string):
        if path_string not in self.path_readings:
            print("Error: Path reading not yet initialized, discarding reading")
            return False
        return True

    def is_complete(self):
        if len(self.path_readings) < Reading.EXPECTED_PATH_COUNT:
            return False
        for _, path_reading in self.path_readings.items():
            if not path_reading.is_complete():
                return False
        return True
        # return "3_to_3" in self.path_readings and self.path_readings["3_to_3"].is_complete()

    # accessors to add readings
    def add_path_reading(self, path_string, datetime_obj):
        if path_string in self.path_readings:
            print("Warn: path already tracked in this reading; overwriting!")
        if len(self.path_readings) >= Reading.EXPECTED_PATH_COUNT:
            print("Warn: Adding more paths than expected. Continuing anyways")
        self.path_readings[path_string] = PathReading(path_string, datetime_obj)

    def set_freq(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].set_freq(val)

    def set_calres(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].set_calres(val)

    def set_tof(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].set_tof(val)

    def add_real(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].add_real(val)

    def add_im(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].add_im(val)

    def add_mag(self, path_string, val):
        if not self.path_initialized(path_string):
            return
        self.path_readings[path_string].add_mag(val)

