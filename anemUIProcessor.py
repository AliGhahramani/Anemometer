from __future__ import unicode_literals

from collections import deque, defaultdict
import numpy as np
from anemUIWindow import *


# Processes input as streamed in by anemUI.py, and spawns a UI thread for this anemometer
class AnemometerProcessor:
    def __init__(self, anemometer_id, is_duct, data_dump_func, calibration_period=10,
                 include_calibration=False, phase_only=True):
        self.anemometer_id = anemometer_id
        self.is_duct = is_duct
        self.data_dump_func = data_dump_func
        self.calibration_period = calibration_period
        self.include_calibration = include_calibration
        self.phase_only = phase_only
        self.is_calibrating = True
        self.aw = None
        self.start_time = time.time()
        self.paths = []
        self.median_window_size = 5
        self.past_5_velocity_magnitudes = None  # tracked for room anemometer, to see if we should artificially zero theta and phi for graph readability

        if is_duct:
            self.paths = [(3, 1), (0, 1), (0, 2), (3, 2)]
        else:
            self.paths = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            self.past_5_velocity_magnitudes = deque(maxlen=5)
        l = len(self.paths)

        # Graph buffers, containing data to be added to the graphs
        self.toggle_graph_buffer = [[] for i in range(0, l)]       # list of (# paths) lists. Each list holds a tuple in form (timestamp, (rel_a_b, rel_b_a, abs_a_b, rel_vel)) that has yet to be graphed on a toggle-able graph
        self.general_graph_buffer = [[] for i in range(0, l)]      # Same behavior as toggle_graph_buffer, different contents (duct: (timestamp, temp) per path, room: timestamp, and one of vx, vy, vz, m, theta, phi)
        self.toggle_graph_buffer_med = [[] for i in range(0, l)]   # medians
        self.general_graph_buffer_med = [[] for i in range(0, l)]  # medians

        # Data history
        self.relative_data = [[] for i in range(0, l)]    # Same tuple data as toggle_graph_buffer, but for all data
        self.general_data = [[] for i in range(0, l)]     # Same tuple data as general_graph_buffer, but for all data

        if phase_only:
            self._prev_rel_phase = {}  # {(src, dst) : number}
            self._prev_abs_phase = {}  # {(src, dst) : number}
            self._cur_abs_phase = {}  # {(src, dst) : number}
            self._calibration_phases = {}  # {(src, dst) : []}, used to hold rel phases during calibration period
            self._calibration_indices = {}  # {src, dst): []}, used to hold indices of max-2 during calibration period
            self._calibrated_index = {}  # {src, dst): index}
        else:
            self._calibration_indices = defaultdict(list)
            self._calibration_magnitudes = defaultdict(list)
            self._calibration_phases = defaultdict(list)

            self._calibrated_index = {}
            self._calibrated_magnitude = {}
            self._calibrated_phase = {}

        # Added by Yannan
        self.past_5_counter = [0, 0, 0, 0, 0, 0]

        self.past_vx = []
        self.past_vy = []
        self.past_vz = []
        #duct is 4
        # End added by Yannan
    def generate_window(self):
        self.aw = ApplicationWindow(None, self, self.is_duct, self.anemometer_id, self.paths)
        return self.aw

    def dump_raw_data(self):
        self.data_dump_func()

    def process_reading(self, reading):
        if self.phase_only:
            # print("phase_only!!!!!!!!!!!!!!!")
            self._process_reading_phase(reading)
        else:
            # print("magnitude!!!!!!!!!!!!")
            self._process_reading_magnitude(reading)

    # this does not work with new reading form
    # Calculate the relative phases, velocities, and other values for this reading using MAGNITUDE
    def _process_reading_magnitude(self, reading):
        print("Magnitude algorithm not implemented")
        return

        print(self.anemometer_id, "processing reading")

        # For each path, calculate absolute phase of reading 2 before max magnitude and magnitude increase per wave
        abs_phase = {}  # {src_to_dst : phase in degrees}
        magnitude = {}  # {src_to_dst : magnitude at index(maximum) - 2}
        magnitude_per_wave = {}  # {src_to_dst : magnitude increase per wave}
        timestamp = reading.timestamp - self.start_time
        for path_string, path_reading in reading.path_readings.items():
            if self.is_calibrating:
                # Find appropriate index (and track it for calibration)
                max_mag = -1
                max_ind = -1
                for i in range(len(path_reading.mag)):
                    if path_reading.mag[i] > max_mag:
                        max_mag = path_reading.mag[i]
                        max_ind = i
                index = max_ind - 2
                if max_ind != -1:
                    self._calibration_indices[path_string].append(index)
                else:  # This should really never happen for a non-self loop, or something is terribly wrong.
                    print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ",
                          path_string, "\nmagnitudes", path_reading.mag)
            else:
                index = self._calibrated_index[path_string]

            if index < 0:
                print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ",
                      path_string, "magnitudes", path_reading.mag)
                abs_phase[path_string] = 0
                magnitude[path_string] = 0
                magnitude_per_wave[path_string] = 0
            else:
                # arctan2 returns 4 quadrant arctan in (-pi, pi)
                abs_phase[path_string] = np.arctan2(path_reading.im[index], path_reading.real[index]) * 180 / np.pi
                if abs_phase[path_string] < 0:
                    abs_phase[path_string] += 360
                magnitude[path_string] = path_reading.mag[index]
                magnitude_per_wave[path_string] = (path_reading.mag[index + 1] - path_reading.mag[index])/8

            if path_string == "0_to_1":
                print("using index ", index, "from mags", path_reading.mag, ", got ", magnitude[path_string])



        # If calibrating, track phase and magnitude
        if self.is_calibrating:
            for path_string, p in abs_phase.items():
                self._calibration_phases[path_string].append(p)
            for path_string, m in magnitude.items():
                self._calibration_magnitudes[path_string].append(m)

            # finish calibration if applicable
            if len(self._calibration_phases) == 16 and len(self._calibration_phases["0_to_1"]) >= self.calibration_period:
                self.is_calibrating = False

                for path_string, index_list in self._calibration_indices.items():
                    index_list = [i for i in index_list if i >= 0]  # ignore all invalid indices
                    index_mode = 0
                    if len(index_list) > 0:  # possible if max is usually at index 0 or 1
                        index_mode = max(set(index_list), key=index_list.count)
                    self._calibrated_index[path_string] = index_mode
                    print(self.anemometer_id, "index[", path_string, "] = ", index_list, ", mode =", index_mode)

                for path_string, phase_list in self._calibration_phases.items():
                    # TODO: should really not include 0 phases that correspond to invalid indices
                    phase_list = [i for i in phase_list if i != 0]
                    m = np.mean(phase_list)
                    sd = np.std(phase_list)
                    if sd == 0:
                        continue
                    s = 0
                    count = 0
                    for v in phase_list:
                        if abs(v - m) / sd < 2:
                            s += v
                            count += 1
                    if count != 0:
                        self._calibrated_phase[path_string] = s / count
                    else:
                        self._calibrated_phase[path_string] = 0

                for path_string, mag_list in self._calibration_magnitudes.items():
                    mag_list = [i for i in mag_list if i > 0]  # ignore all invalid magnitudes
                    self._calibrated_magnitude[path_string] = np.mean(mag_list)

            # If include_calibration, also graph the phases during calibration period.
            if self.include_calibration and abs_phase != {}:
                for i in range(len(self.paths)):
                    a, b = self.paths[i]
                    abs_phase_ab = abs_phase[str(a) + "_to_" + str(b)]
                    self.toggle_graph_buffer[i].append((timestamp, (0, 0, abs_phase_ab, 0)))
        # If not calibrating, calculate pairwise velocities.
        else:
            # For each path, calculate relative phase using absolute phase and magnitude
            rel_phase = {}  # {src_to_dst : phase in degrees}
            for path_string in abs_phase.keys():
                p = abs_phase[path_string]
                m = magnitude[path_string]
                mpw = magnitude_per_wave[path_string]
                p_c = self._calibrated_phase[path_string]
                m_c = self._calibrated_magnitude[path_string]
                max_wave_change = 3
                m_compare = []

                for i in range(-max_wave_change, max_wave_change+1):
                    v = m_c + ((p - p_c)/360 + i) * mpw # try + and - ...
                    m_compare.append(v)

                i = (np.abs(np.subtract(m_compare, [m] * len(m_compare)))).argmin() - max_wave_change
                rel_phase[path_string] = i * 360 + p - p_c
                if path_string == "0_to_1":
                    print(path_string)
                    print(m_compare)
                    print("p ", p, ", m ", m, ", mpw ", mpw, ", p_c ", p_c, ", m_c ", m_c, ", i ", i, ", rel_phase ",
                          rel_phase[path_string])

            all_v_rel = []
            for i in range(len(self.paths)):
                # TODO: scale distances properly for duct
                d = 0.1875
                if not self.is_duct:
                    d = 0.06
                a, b = self.paths[i]
                phase_ab = rel_phase[str(a) + "_to_" + str(b)]
                phase_ba = rel_phase[str(b) + "_to_" + str(a)]
                abs_phase_ab = abs_phase[str(a) + "_to_" + str(b)]
                tof_ab = phase_ab / (360 * 180000)  # khz?
                tof_ba = phase_ba / (360 * 180000)

                v_ab = d / (d / 343 + tof_ab)
                v_ba = d / (d / 343 + tof_ba)
                v_rel = (v_ab - v_ba) / 2 / np.cos(45 * np.pi / 180.)
                if self.is_duct:
                    # v_rel = -v_rel  # sign is flipped for four path for other anemometer
                    v_ab = d / (d / 344 - tof_ab)
                    v_ba = d / (d / 344 - tof_ba)
                    avg = (v_ab + v_ba) / 2
                    temp = avg * avg / 400 - 273.15
                    # print(temp)
                    # TODO: These groups of appends should really be put in a helper function for readability.
                    self.general_graph_buffer[i].append((timestamp, temp))
                    self.general_data[i].append((timestamp, temp))

                self.toggle_graph_buffer[i].append((timestamp, (phase_ab, phase_ba, abs_phase_ab, v_rel)))
                self.relative_data[i].append((timestamp, (phase_ab, phase_ba, abs_phase_ab, v_rel)))
                all_v_rel.append(v_rel)

            # For room anemometer, also calculate vx, vy, vz, m, theta, phi. Weighted assuming node 1 at bottom.
            if not self.is_duct:
                sin30 = np.sin(np.pi / 6)
                sin60 = np.sin(np.pi / 3)
                vx_weight = [sin30 * sin30, sin60, 0, sin30, -sin30 * sin30, -sin60]
                vy_weight = [sin60 * sin30, sin30, 1, 0, sin60 * sin30, sin30]
                vz_weight = [-sin60, 0, 0, sin60, sin60, 0]
                vx = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vx_weight) if i[1] != 0])) / tnx				
                vy = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vy_weight) if i[1] != 0])) / tny
                vz = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vz_weight) if i[1] != 0])) / 3
 
                m = np.sqrt(vx * vx + vy * vy + vz * vz)
                avg_m = 0 if len(self.past_5_velocity_magnitudes) == 0 else sum(self.past_5_velocity_magnitudes) / len(
                    self.past_5_velocity_magnitudes)
                theta = np.arctan2(vy, vx) * 180 / np.pi if avg_m > 1 else 0
                phi = np.arcsin(vz / m) * 180 / np.pi if avg_m > 1 else 0
                self.general_graph_buffer[0].append((timestamp, vx))
                self.general_graph_buffer[1].append((timestamp, vy))
                self.general_graph_buffer[2].append((timestamp, vz))
                self.general_graph_buffer[3].append((timestamp, m))
                self.general_graph_buffer[4].append((timestamp, theta))
                self.general_graph_buffer[5].append((timestamp, phi))

                self.general_data[0].append((timestamp, vx))
                self.general_data[1].append((timestamp, vy))
                self.general_data[2].append((timestamp, vz))
                self.general_data[3].append((timestamp, m))
                self.general_data[4].append((timestamp, theta))
                self.general_data[5].append((timestamp, phi))
                self.past_5_velocity_magnitudes.append(m)

            # Calculate median over window and put in buffer
            if len(self.general_data[0]) >= self.median_window_size:
                for i in range(len(self.paths)):
                    y_med = np.median([x[1] for x in self.general_data[i][-self.median_window_size:]])
                    x_med = np.median([x[0] for x in self.general_data[i][-self.median_window_size:]])
                    self.general_graph_buffer_med[i].append((x_med, y_med))

                    y_med = np.median([x[1][3] for x in self.relative_data[i][-self.median_window_size:]]) # Median relative velocity
                    x_med = np.median([x[0] for x in self.relative_data[i][-self.median_window_size:]])
                    self.toggle_graph_buffer_med[i].append((x_med, y_med))


    def _process_reading_phase(self, reading):
        # For each path, calculate absolute phase of reading 2 before max magnitude
        print(self.anemometer_id, "processing reading")
        num_sensors = reading.num_sensors
        data_len = 4

        # figure out where to read from, and calculate absolute phase
        next_abs_phase = {}  # {(src, dst) : phase in degrees}
        read_index = {}  # {(src, dst) : index offset from this reading's max on which to do calculations }
        # timestamp = reading.timestamp - self.start_time
        timestamp = time.time() - self.start_time
        for src in range(0, num_sensors):
            for dst in range(0, num_sensors):
                if src == dst:
                    continue # don't process anything for self-loops
                cur_max_index = reading.get_max_index(src, dst)

                if self.is_calibrating:
                    if (src, dst) not in self._calibration_indices:
                        self._calibration_indices[(src, dst)] = []
                    self._calibration_indices[(src, dst)].append(cur_max_index)
                    offset_from_max = -2
                    read_index[(src, dst)] = data_len + offset_from_max - 1
                else:
                    max_index = self._calibrated_index[(src, dst)]
                    offset_from_max = -2 + (max_index - cur_max_index)
                    read_index[(src, dst)] = data_len + offset_from_max - 1
                    # print("cur_max_index", cur_max_index, "max_index", max_index, "read_index", read_index[(src, dst)])

                if read_index[(src, dst)] < 0 or read_index[(src, dst)] >= data_len:
                    print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ", (src, dst),
                          ", index out of range.\n\t", "cur_max_index", cur_max_index,
                          "max_index", self._calibrated_index[(src, dst)],
                          "read_index", read_index[(src, dst)])
                    next_abs_phase[(src, dst)] = 0
                else:
                    i = reading.get_imaginary(src, dst)[read_index[(src, dst)]]
                    q = reading.get_real(src, dst)[read_index[(src, dst)]]
                    next_abs_phase[(src, dst)] = np.arctan2(i, q) * 180 / np.pi



        # Infer current reading's relative phase, using previous relative phase, and previous, current,
        # and next absolute phases. See README
        cur_rel_phase = {}
        outlier_range = 80  # if difference of this reading from previous reading is within X degrees of 180 degrees, can assume it's (dangerous) reading error
        jerk_limit = 120  # tolerates at most an X difference in phase deltas
        if self._cur_abs_phase == {}:
            # This is the first reading; the buffer is still filling.
            pass
        elif self._prev_abs_phase == {}:
            # This is the second reading; the buffer is still filling. The 'current' reading has no relative phase to compare to.
            for (src, dst), abs_phase in self._cur_abs_phase.items():
                cur_rel_phase[(src, dst)] = abs_phase
        else:
            # All other readings. Infer cur_rel_phase from prev/cur/next_abs_phase, and prev_rel_phase
            deltas = []
            for (src, dst) in self._cur_abs_phase:
                cur_delta = self._cur_abs_phase[(src, dst)] - self._prev_abs_phase[(src, dst)]
                next_delta = next_abs_phase[(src, dst)] - self._cur_abs_phase[(src, dst)]
                cur_sign = 1 if cur_delta >= 0 else -1
                next_sign = 1 if next_delta >= 0 else -1
                cur_wrapped_delta = cur_delta if abs(cur_delta) < 180 else -1 * cur_sign * (360 - abs(cur_delta))
                next_wrapped_delta = next_delta if abs(next_delta) < 180 else -1 * next_sign * (360 - abs(next_delta))

                if ((180 - abs(cur_wrapped_delta)) <= outlier_range) or \
                        (abs(cur_wrapped_delta - next_wrapped_delta) > jerk_limit):
                    # Outlier; drop this point
                    cur_rel_phase[(src, dst)] = self._prev_rel_phase[(src, dst)]
                    self._cur_abs_phase[(src, dst)] = self._prev_abs_phase[(src, dst)]
                    deltas.append((0, (src, dst)))
                else:
                    cur_rel_phase[(src, dst)] = self._prev_rel_phase[(src, dst)] + cur_wrapped_delta
                    deltas.append((cur_wrapped_delta, (src, dst)))

            # filter out readings that look like outliers, as compared to other paths
            deltas.sort(key=lambda x: x[0])
            delta_low = deltas[1][0] - deltas[0][0]
            delta_low_compare = deltas[2][0] - deltas[1][0]
            delta_high = deltas[-1][0] - deltas[-2][0]
            delta_high_compare = deltas[-2][0] - deltas[-3][0]
            if delta_low > 2 * delta_low_compare and delta_low > 30:
                # lowest reading is an outlier
                (src, dst) = deltas[0][1]
                cur_rel_phase[(src, dst)] = self._prev_rel_phase[(src, dst)]
                self._cur_abs_phase[(src, dst)] = self._prev_abs_phase[(src, dst)]
            if delta_high > 2 * delta_high_compare and delta_high > 30:
                # highest reading is an outlier
                (src, dst) = deltas[-1][1]
                cur_rel_phase[(src, dst)] = self._prev_rel_phase[(src, dst)]
                self._cur_abs_phase[(src, dst)] = self._prev_abs_phase[(src, dst)]


        # If calibrating, track phase
        if self.is_calibrating:
            for (src, dst), rel_phase in cur_rel_phase.items():
                if (src, dst) not in self._calibration_phases:
                    self._calibration_phases[(src, dst)] = []
                self._calibration_phases[(src, dst)].append(rel_phase)

            # finish calibration if applicable
            if len(self._calibration_phases) == num_sensors * (num_sensors - 1) and len(
                    self._calibration_phases[(0, 1)]) >= self.calibration_period:
                self.is_calibrating = False
                for (src, dst), phase_list in self._calibration_phases.items():
                    print("I am ", self.anemometer_id, " with path ", (src, dst), ", phases: ", phase_list)
                    m = np.mean(phase_list)
                    sd = np.std(phase_list)
                    if sd == 0:
                        continue
                    s = 0
                    count = 0
                    for v in phase_list:
                        if abs(v - m) / sd < 2:
                            s += v
                            count += 1
                    if count != 0:
                        cur_rel_phase[(src, dst)] -= s / count

                for (src, dst), index_list in self._calibration_indices.items():
                    index_list = [i for i in index_list if i >= 0]  # ignore all invalid indices
                    index_mode = 0
                    if len(index_list) > 0:  # possible if max is usually at index 0 or 1
                        index_mode = max(set(index_list), key=index_list.count)
                    self._calibrated_index[(src, dst)] = index_mode
                    print(self.anemometer_id, "index[", (src, dst), "] = ", index_list, ", mode =", index_mode)

            # If include_calibration, also graph the phases during calibration period.
            if self.include_calibration and cur_rel_phase != {}:
                for i in range(len(self.paths)):
                    (src, dst) = self.paths[i]
                    phase_ab = cur_rel_phase[(src, dst)]
                    phase_ba = cur_rel_phase[(src, dst)]
                    abs_phase_ab = self._cur_abs_phase[(src, dst)]
                    self.toggle_graph_buffer[i].append((timestamp, (phase_ab, phase_ba, abs_phase_ab, 0)))

        # If not calibrating, calculate pairwise velocities.
        else:
            all_v_rel = []
            for i in range(len(self.paths)):
                # TODO: scale distances properly for duct
                d = 0.1875
                if not self.is_duct:
                    d = 0.06
                (src, dst) = self.paths[i]
                phase_ab = cur_rel_phase[(src, dst)]
                phase_ba = cur_rel_phase[(dst, src)]
                abs_phase_ab = self._cur_abs_phase[(src, dst)]
                tof_ab = phase_ab / (360 * 180000)  # khz?
                tof_ba = phase_ba / (360 * 180000)

                v_ab = d / (d / 343 + tof_ab)
                v_ba = d / (d / 343 + tof_ba)
                v_rel = (v_ab - v_ba) / 2 / np.cos(45 * np.pi / 180.)
                if self.is_duct:
                    # v_rel = -v_rel  # sign is flipped for four path for other anemometer
                    v_ab = d / (d / 344 - tof_ab)
                    v_ba = d / (d / 344 - tof_ba)
                    avg = (v_ab + v_ba) / 2
                    temp = avg * avg / 400 - 273.15
                    # print(temp)
                    self.general_graph_buffer[i].append((timestamp, temp))
                    self.general_data[i].append((timestamp, temp))

                self.toggle_graph_buffer[i].append((timestamp, (phase_ab, phase_ba, abs_phase_ab, v_rel)))
                self.relative_data[i].append((timestamp, (phase_ab, phase_ba, abs_phase_ab, v_rel)))
                all_v_rel.append(v_rel)

            # Added by Yannan
            self.velocity_outlier_filter(all_v_rel, cur_rel_phase)
            # End added by Yannan

            # For room anemometer, also calculate vx, vy, vz, m, theta, phi. Weighted assuming node 1 at bottom.
            if not self.is_duct:
                sin30 = np.sin(np.pi / 6)
                sin60 = np.sin(np.pi / 3)
                vx_weight = [sin30 * sin30, sin60, 0, sin30, -sin30 * sin30, -sin60]
                vy_weight = [sin60 * sin30, sin30, 1, 0, sin60 * sin30, sin30]
                vz_weight = [-sin60, 0, 0, sin60, sin60, 0]
                #vx_weight = [0, sin60, 0, 0, 0, -sin60]
                #vy_weight = [0, sin30, 1, 0, 0, sin30]
                #vz_weight = [-sin60, 0, 0, sin60, sin60, 0]
                tnx = 5 # 6 if 15degree coordinate system
                tny = 5 # 6 if 15degree coordinate system
                tnz = [] 
                if ( 0 > abs(all_v_rel[1]) / all_v_rel[2] > - 0.5 and 0 > abs(all_v_rel[1])/all_v_rel[5] > -0.5): 
                    vx_weight = [0, sin60*0.85, 0, 0, 0 , -sin60*0.85]
                    vy_weight = [0, sin30*0.85,1*0.85, 0, 0, sin30*0.85]
                    tnx = 2
                    tny = 3
                    tnz = [0,3]
                if (0>abs(all_v_rel[2]) / all_v_rel[1] > -0.5 and 0 < abs(all_v_rel[2]) /all_v_rel[5] < 0.5): 
                    vx_weight = [0, sin60*0.85, 0, 0, 0 , -sin60*0.85]
                    vy_weight = [0, sin30*0.85,1*0.85, 0, 0, sin30*0.85]
                    tnx = 2
                    tny = 3
                    tnz = [0,4]
                if (0<abs(all_v_rel[5])/ all_v_rel[1] < 0.5 and 0<abs(all_v_rel[5])/all_v_rel[2] < 0.5): 
                    vx_weight = [0, sin60*0.85, 0, 0, 0 , -sin60*0.85]
                    vy_weight = [0, sin30*0.85,1*0.8, 0, 0, sin30*0.85]
                    tnx = 2
                    tny = 3
                    tnz = [3,4]
            #     
                if ( 0 < abs(all_v_rel[0] / all_v_rel[3]) <  0.5 and 0 < abs(all_v_rel[0]/all_v_rel[4]) < 0.5): 
                    vx_weight = [sin30 * sin30*1.1, 0, 0, sin30*1.1, -sin30*sin30*1.1 , 0]
                    vy_weight = [sin60 * sin30*1.1, 0,0, 0, sin60*sin30*1.1, 0]
                    tnx = 3
                    tny = 2
                if ( 0 < abs(all_v_rel[3] / all_v_rel[0]) <  0.5 and 0 < abs(all_v_rel[3]/all_v_rel[4]) < 0.5): 
                    vx_weight = [sin30 * sin30*1.1, 0, 0, sin30*1.1, -sin30*sin30 *1.1, 0]
                    vy_weight = [sin60 * sin30*1.1, 0,0, 0, sin60*sin30*1.1, 0]
                    tnx = 3
                    tny = 2
                if ( 0 < abs(all_v_rel[4] / all_v_rel[0]) <  0.5 and 0 < abs(all_v_rel[4]/all_v_rel[0]) < 0.5): 
                    vx_weight = [sin30 * sin30*1.1, 0, 0, sin30*1.1, -sin30*sin30*1.1 , 0]
                    vy_weight = [sin60 * sin30*1.1, 0,0, 0, sin60*sin30*1.1, 0]
                    tnx = 3
                    tny = 2
                vx = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vx_weight) if i[1] != 0])) / tnx				
                vy = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vy_weight) if i[1] != 0])) / tny
                vz = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vz_weight) if i[1] != 0])) / 3
                if (len(tnz)>1): 
                    vp = np.sqrt(vx * vx + vy * vy) 
                    #vz = (all_v_rel[tnz[0]]/vz_weight[tnz[0]]+all_v_rel[tnz[1]]/vz_weight[tnz[1]])/2 
                #vx = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vx_weight) if i[1] != 0])) / 5
                #vy = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vy_weight) if i[1] != 0])) / 5
                #vz = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vz_weight) if i[1] != 0])) / 3
                m = np.sqrt(vx * vx + vy * vy + vz * vz)
                avg_m = 0 if len(self.past_5_velocity_magnitudes) == 0 else sum(self.past_5_velocity_magnitudes) / len(
                    self.past_5_velocity_magnitudes)
                theta = np.arctan2(vy, vx) * 180 / np.pi if avg_m > 0.5 else 0


                # modify m for phi --> 
                # keep track of past 10 vx, vy, vz
                # average of past 10 vx, average of past 10 vy, average of past 10 vz, and then calculate 
                # a temporary m for phi with this value. 
                # do only when sqrt(vx^2 + vy^2) < 0.5
                # for visualization, to cancel out noise and approach 0
                # Begin added by Yannan
                if len(self.past_vx) < 10:
                    self.past_vx.append(vx)
                    self.past_vy.append(vy)
                    self.past_vz.append(vz)
                else:
                    self.past_vx = self.past_vx[1:] + [vx]
                    self.past_vy = self.past_vy[1:] + [vy]
                    self.past_vz = self.past_vz[1:] + [vz]


                if abs(vx) < 0.5 and abs(vy) < 0.5 and abs(vz) < 0.5:
                    temp_vx = mean(self.past_vx)
                    temp_vy = mean(self.past_vy)
                    temp_vz = mean(self.past_vz)
                    m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(temp_vz, 2))

                if len(self.past_vx) < 10:
                    phi = np.arcsin(vz / m) * 180 / np.pi if avg_m > 0.5 else 0
                else:                    
                    if abs(vz) < 0.5 and np.sqrt(pow(vx, 2) + pow(vy, 2)) < 0.5:
                        temp_vx = mean(self.past_vx)
                        temp_vy = mean(self.past_vy)
                        temp_vz = mean(self.past_vz)
                        temp_m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(temp_vz, 2))
                        phi = np.arcsin(temp_vz / temp_m) * 180 / np.pi if avg_m > 0.5 else 0
                    elif np.sqrt(pow(vx, 2) + pow(vy, 2)) < 0.5: 
                        temp_vx = mean(self.past_vx)
                        temp_vy = mean(self.past_vy)
                        temp_m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(vz, 2))
                        phi = np.arcsin(vz / temp_m) * 180 / np.pi if avg_m > 0.5 else 0
                    elif abs(vz) < 0.5:
                        temp_vz = mean(self.past_vz)
                        temp_m = np.sqrt(pow(vx, 2) + pow(vy, 2) + pow(temp_vz, 2))
                        phi = np.arcsin(temp_vz / temp_m) * 180 / np.pi if avg_m > 0.5 else 0
                    else:
                    # End added by Yannan
                        #original
                        phi = np.arcsin(vz / m) * 180 / np.pi if avg_m > 0.5 else 0

                self.general_graph_buffer[0].append((timestamp, vx))
                self.general_graph_buffer[1].append((timestamp, vy))
                self.general_graph_buffer[2].append((timestamp, vz))
                self.general_graph_buffer[3].append((timestamp, m))
                self.general_graph_buffer[4].append((timestamp, theta))
                self.general_graph_buffer[5].append((timestamp, phi))

                self.general_data[0].append((timestamp, vx))
                self.general_data[1].append((timestamp, vy))
                self.general_data[2].append((timestamp, vz))
                self.general_data[3].append((timestamp, m))
                self.general_data[4].append((timestamp, theta))
                self.general_data[5].append((timestamp, phi))
                self.past_5_velocity_magnitudes.append(m)

            if len(self.general_data[0]) >= self.median_window_size:
                for i in range(len(self.paths)):
                    y_med = np.median([x[1] for x in self.general_data[i][-self.median_window_size:]])
                    x_med = np.median([x[0] for x in self.general_data[i][-self.median_window_size:]])
                    self.general_graph_buffer_med[i].append((x_med, y_med))

                    y_med = np.median([x[1][3] for x in self.relative_data[i][-self.median_window_size:]]) # Median relative velocity
                    x_med = np.median([x[0] for x in self.relative_data[i][-self.median_window_size:]])
                    self.toggle_graph_buffer_med[i].append((x_med, y_med))

        # Rotate over the readings to prepare for the next reading.
        self._prev_rel_phase = cur_rel_phase
        self._prev_abs_phase = self._cur_abs_phase
        self._cur_abs_phase = next_abs_phase
    # Begin added by Yannan
    # place check after for loop, all values have been appended
    # self.toggle_graph_buffer[i].appends v_rel as well --> toggle canvas issue, second priority?
    # otherwise, also need to pass this in and edit somehow (with temporary list, then extend + append possibly)
    def velocity_outlier_filter(self, all_v_rel, cur_rel_phase):
        outlier_index = -1
        for i in range(len(all_v_rel)):
            # Take mean of all values except value in question (not averaging whole thing?)
            rest_of_lst = all_v_rel[0:i] + all_v_rel[i+1:]
            avg = mean([abs(val) for val in rest_of_lst])
            condition = avg < 0.5 and abs(all_v_rel[i]) > (avg + 1)
            if condition:
                outlier_index = i
                continue            #assumes only one outlier
        if outlier_index != -1:
            # counter for each index 
            self.past_5_counter[outlier_index] += 1
            # > or >=?
            if self.past_5_counter[outlier_index] >= 20:
                # change cur_rel_phase, only change for incorrect index; mean of the rest of indices
                (src, dst) = self.paths[outlier_index]
                (dst, src) = self.paths[outlier_index]
                new_val = (sum(cur_rel_phase.values()) - cur_rel_phase[(src, dst)] - cur_rel_phase[(dst, src)])/(len(cur_rel_phase)-2)
                cur_rel_phase[(src, dst)] = new_val
                cur_rel_phase[(dst, src)] = new_val
                # all_v_rel[outlier_index] = avg
                self.past_5_counter[outlier_index] = 0
        else:
            # reset counter if condition is not seen? intermittent issues?
            for count in self.past_5_counter:
                count = 0

    def update_medians(self, median_window_size, is_toggle_graph, index):
        # Todo: small bug where buffer may still hold a few medians using the old median window size. 
        x_medians = []
        y_medians = []
        if is_toggle_graph:
            if len(self.relative_data[index]) < median_window_size:
                return ([], [])
            for i in range(len(self.relative_data[index]) - median_window_size + 1):
                x_medians.append(np.median([x[0] for x in self.relative_data[index][i : i + median_window_size]]))
                y_medians.append(np.median([x[1][3] for x in self.relative_data[index][i : i + median_window_size]]))
        else:
            if len(self.general_data[index]) < median_window_size:
                return ([], [])
            for i in range(len(self.general_data[index]) - median_window_size + 1):
                x_medians.append(np.median([x[0] for x in self.general_data[index][i : i + median_window_size]]))
                y_medians.append(np.median([x[1] for x in self.general_data[index][i : i + median_window_size]]))
        return (x_medians, y_medians)

def mean(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)
    # End added by Yannan
# old implementation with Readings
    # Calculate the relative phases, velocities, and other values for this reading using PHASE ONLY
    # def _process_reading_phase(self, reading):
    #     # For each path, calculate absolute phase of reading 2 before max magnitude
    #     print(self.anemometer_id, "processing reading")
    #     next_abs_phase = {}  # {src_to_dst : phase in degrees}
    #     for path_string, path_reading in reading.path_readings.items():
    #         if self.is_calibrating:
    #             max_mag = -1
    #             max_ind = -1
    #             for i in range(len(path_reading.mag)):
    #                 if path_reading.mag[i] > max_mag:
    #                     max_mag = path_reading.mag[i]
    #                     max_ind = i
    #             index = max_ind - 2
    #             if path_string not in self._calibration_indices:
    #                 self._calibration_indices[path_string] = []
    #             if max_ind != -1:
    #                 self._calibration_indices[path_string].append(index)
    #             else:  # This should really never happen, or something is terribly wrong.
    #                 print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ",
    #                       path_reading.path_string, "\nmagnitudes", path_reading.mag)
    #         else:
    #             index = self._calibrated_index[path_string]
    #
    #         if index < 0:
    #             print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ",
    #                   path_reading.path_string, "magnitudes", path_reading.mag)
    #             next_abs_phase[path_reading.path_string] = 0
    #         else:
    #             # arctan2 returns 4 quadrant arctan in (-pi, pi)
    #             next_abs_phase[path_reading.path_string] = np.arctan2(path_reading.im[index],
    #                                                               path_reading.real[index]) * 180 / np.pi
    #
    #     # Infer current reading's relative phase, using previous relative phase, and previous, current,
    #     # and next absolute phases. See README
    #     cur_rel_phase = {}
    #     outlier_range = 80  # if difference of this reading from previous reading is within X degrees of 180 degrees, can assume it's (dangerous) reading error
    #     jerk_limit = 120  # tolerates at most an X difference in phase deltas
    #     if self._cur_abs_phase == {}:
    #         # This is the first reading; the buffer is still filling.
    #         pass
    #     elif self._prev_abs_phase == {}:
    #         # This is the second reading; the buffer is still filling. The 'current' reading has no relative phase to compare to.
    #         for path_string, abs_phase in self._cur_abs_phase.items():
    #             cur_rel_phase[path_string] = abs_phase
    #     else:
    #         # All other readings. Infer cur_rel_phase from prev/cur/next_abs_phase, and prev_rel_phase
    #         deltas = []
    #         for path_string in self._cur_abs_phase:
    #             cur_delta = self._cur_abs_phase[path_string] - self._prev_abs_phase[path_string]
    #             next_delta = next_abs_phase[path_string] - self._cur_abs_phase[path_string]
    #             cur_sign = 1 if cur_delta >= 0 else -1
    #             next_sign = 1 if next_delta >= 0 else -1
    #             cur_wrapped_delta = cur_delta if abs(cur_delta) < 180 else -1 * cur_sign * (360 - abs(cur_delta))
    #             next_wrapped_delta = next_delta if abs(next_delta) < 180 else -1 * next_sign * (360 - abs(next_delta))
    #
    #             if ((180 - abs(cur_wrapped_delta)) <= outlier_range) or \
    #                     (abs(cur_wrapped_delta - next_wrapped_delta) > jerk_limit):
    #                 # Outlier; drop this point
    #                 cur_rel_phase[path_string] = self._prev_rel_phase[path_string]
    #                 self._cur_abs_phase[path_string] = self._prev_abs_phase[path_string]
    #                 deltas.append((0, path_string))
    #             else:
    #                 cur_rel_phase[path_string] = self._prev_rel_phase[path_string] + cur_wrapped_delta
    #                 deltas.append((cur_wrapped_delta, path_string))
    #
    #         # filter out readings that look like outliers, as compared to other paths
    #         deltas.sort(key=lambda x: x[0])
    #         delta_low = deltas[1][0] - deltas[0][0]
    #         delta_low_compare = deltas[2][0] - deltas[1][0]
    #         delta_high = deltas[-1][0] - deltas[-2][0]
    #         delta_high_compare = deltas[-2][0] - deltas[-3][0]
    #         if delta_low > 2 * delta_low_compare and delta_low > 30:
    #             # lowest reading is an outlier
    #             path_string = deltas[0][1]
    #             cur_rel_phase[path_string] = self._prev_rel_phase[path_string]
    #             self._cur_abs_phase[path_string] = self._prev_abs_phase[path_string]
    #         if delta_high > 2 * delta_high_compare and delta_high > 30:
    #             # highest reading is an outlier
    #             path_string = deltas[-1][1]
    #             cur_rel_phase[path_string] = self._prev_rel_phase[path_string]
    #             self._cur_abs_phase[path_string] = self._prev_abs_phase[path_string]
    #
    #     # If calibrating, track phase
    #     if self.is_calibrating:
    #         for path_string, rel_phase in cur_rel_phase.items():
    #             if path_string not in self._calibration_phases:
    #                 self._calibration_phases[path_string] = []
    #             self._calibration_phases[path_string].append(rel_phase)
    #
    #         # finish calibration if applicable
    #         if len(self._calibration_phases) == 16 and len(self._calibration_phases["0_to_1"]) >= self.calibration_period:
    #             self.is_calibrating = False
    #             for path_string, phase_list in self._calibration_phases.items():
    #                 print("I am ", self.anemometer_id, " with path ", path_string, ", phases: ", phase_list)
    #                 m = np.mean(phase_list)
    #                 sd = np.std(phase_list)
    #                 if sd == 0:
    #                     continue
    #                 s = 0
    #                 count = 0
    #                 for v in phase_list:
    #                     if abs(v - m) / sd < 2:
    #                         s += v
    #                         count += 1
    #                 if count != 0:
    #                     cur_rel_phase[path_string] -= s/count
    #
    #             for path_string, index_list in self._calibration_indices.items():
    #                 index_list = [i for i in index_list if i >= 0]  # ignore all invalid indices
    #                 index_mode = 0
    #                 if len(index_list) > 0:  # possible if max is usually at index 0 or 1
    #                     index_mode = max(set(index_list), key=index_list.count)
    #                 self._calibrated_index[path_string] = index_mode
    #                 print(self.anemometer_id, "index[", path_string, "] = ", index_list, ", mode =", index_mode)
    #         # If include_calibration, also graph the phases during calibration period.
    #         if self.include_calibration and cur_rel_phase != {}:
    #             for i in range(len(self.paths)):
    #                 a, b = self.paths[i]
    #                 phase_ab = cur_rel_phase[str(a) + "_to_" + str(b)]
    #                 phase_ba = cur_rel_phase[str(b) + "_to_" + str(a)]
    #                 abs_phase_ab = self._cur_abs_phase[str(a) + "_to_" + str(b)]
    #                 self.toggle_graph_buffer_y[i].append((phase_ab, phase_ba, abs_phase_ab, 0))
    #                 self.toggle_graph_buffer_x[i].append(reading.timestamp - self.start_time)
    #     # If not calibrating, calculate pairwise velocities.
    #     else:
    #         all_v_rel = []
    #         for i in range(len(self.paths)):
    #             # TODO: scale distances properly for duct
    #             d = 0.1875
    #             if not self.is_duct:
    #                 d = 0.06
    #             a, b = self.paths[i]
    #             phase_ab = cur_rel_phase[str(a) + "_to_" + str(b)]
    #             phase_ba = cur_rel_phase[str(b) + "_to_" + str(a)]
    #             abs_phase_ab = self._cur_abs_phase[str(a) + "_to_" + str(b)]
    #             tof_ab = phase_ab / (360 * 180000)  # khz?
    #             tof_ba = phase_ba / (360 * 180000)
    #
    #             v_ab = d / (d / 343 + tof_ab)
    #             v_ba = d / (d / 343 + tof_ba)
    #             v_rel = (v_ab - v_ba) / 2 / np.cos(45 * np.pi / 180.)
    #             if self.is_duct:
    #                 # v_rel = -v_rel  # sign is flipped for four path for other anemometer
    #                 v_ab = d / (d / 344 - tof_ab)
    #                 v_ba = d / (d / 344 - tof_ba)
    #                 avg = (v_ab + v_ba) / 2
    #                 temp = avg * avg / 400 - 273.15
    #                 # print(temp)
    #                 self.general_graph_buffer_y[i].append(temp)
    #
    #             self.toggle_graph_buffer_y[i].append((phase_ab, phase_ba, abs_phase_ab, v_rel))
    #             self.toggle_graph_buffer_x[i].append(reading.timestamp - self.start_time)
    #             all_v_rel.append(v_rel)
    #
    #         # For room anemometer, also calculate vx, vy, vz, m, theta, phi. Weighted assuming node 1 at bottom.
    #         if not self.is_duct:
    #             sin30 = np.sin(np.pi / 6)
    #             sin60 = np.sin(np.pi / 3)
    #             vx_weight = [sin30 * sin30, sin60, 0, sin30, -sin30 * sin30, -sin60]
    #             vy_weight = [sin60 * sin30, sin30, 1, 0, sin60 * sin30, sin30]
    #             vz_weight = [-sin60, 0, 0, sin60, sin60, 0]
    #
    #             vx = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vx_weight) if i[1] != 0])) / 5
    #             vy = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vy_weight) if i[1] != 0])) / 5
    #             vz = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vz_weight) if i[1] != 0])) / 3
    #             m = np.sqrt(vx * vx + vy * vy + vz * vz)
    #             avg_m = 0 if len(self.past_5_velocity_magnitudes) == 0 else sum(self.past_5_velocity_magnitudes) / len(
    #                 self.past_5_velocity_magnitudes)
    #             theta = np.arctan2(vy, vx) * 180 / np.pi if avg_m > 1 else 0
    #             phi = np.arcsin(vz / m) * 180 / np.pi if avg_m > 1 else 0
    #             self.general_graph_buffer_y[0].append(vx)
    #             self.general_graph_buffer_y[1].append(vy)
    #             self.general_graph_buffer_y[2].append(vz)
    #             self.general_graph_buffer_y[3].append(m)
    #             self.general_graph_buffer_y[4].append(theta)
    #             self.general_graph_buffer_y[5].append(phi)
    #             self.past_5_velocity_magnitudes.append(m)
    #
    #     # Rotate over the readings to prepare for the next reading.
    #     self._prev_rel_phase = cur_rel_phase
    #     self._prev_abs_phase = self._cur_abs_phase
    #     self._cur_abs_phase = next_abs_phase
