from __future__ import unicode_literals

from collections import deque, defaultdict
import numpy as np
from anemUIWindow import *


# Processes input as streamed in by anemUI.py, and spawns a UI thread for this anemometer
class AnemometerProcessor:
    def __init__(self, anemometer_id, is_duct, data_dump_func, calibration_period,
                 include_calibration=False, use_room_min=True, duct_dist=0.0454):
        self.anemometer_id = anemometer_id
        self.is_duct = is_duct
        self.data_dump_func = data_dump_func
        self.calibration_period = calibration_period
        self.include_calibration = include_calibration
        self.use_room_min = use_room_min
        self.duct_distance = duct_dist

        self.algorithm = 1  # 0: phase only. 1: Temperature guided\
        self.is_calibrating = True
        self.aw = None
        self.start_time = time.time()
        self.paths = []
        self.median_window_size = 5
        self.median_window_size_extended = 5
        self.low_velocity_threshold = 0.2
        self.m_zero = 0.1  # velocities at this speed and below are mapped to 0 m/s
        self.past_5_velocity_magnitudes = None  # tracked for room anemometer, to see if we should artificially zero theta and phi for graph readability

        if is_duct:
            self.paths = [(3, 1), (0, 1), (0, 2), (3, 2)]
        else:
            self.paths = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            self.past_5_velocity_magnitudes = deque(maxlen=5)
        l = len(self.paths)
        num_strip_graphs = 2 if is_duct else 3

        # Graph buffers, containing data to be added to the graphs
        self.toggle_graph_buffer = [[] for _ in range(0, l)]  # list of (# paths) lists. Each list holds a tuple in form (timestamp, (rel_a_b, rel_b_a, abs_a_b, rel_vel)) that has yet to be graphed on a toggle-able graph
        self.general_graph_buffer = [[] for _ in range(0, l)]  # Same behavior as toggle_graph_buffer, different contents (duct: (timestamp, temp) per path, room: timestamp, and one of vx, vy, vz, m, theta, phi)
        self.strip_graph_buffer = [[] for _ in range(0, num_strip_graphs)]  # Same behavior as toggle_graph_buffer, duct: timestamp, and one of speed or temp. room: timestamp, and one of speed, azimuth, or temp
        self.toggle_graph_buffer_med = [[] for _ in range(0, l)]  # medians
        self.general_graph_buffer_med = [[] for _ in range(0, l)]  # medians
        self.strip_graph_buffer_med = [[] for _ in range(0, num_strip_graphs)]
        self.general_graph_buffer_blank = [[] for _ in range (0, l)]
        self.strip_graph_buffer_blank = [[] for _ in range(0, num_strip_graphs)]

        # Data history
        self.relative_data = [[] for _ in range(0, l)]  # Same tuple data as toggle_graph_buffer, but for all data
        self.general_data = [[] for _ in range(0, l)]  # Same tuple data as general_graph_buffer, but for all data
        self.strip_data = [[] for _ in range(0, num_strip_graphs)]
        self.path_vels = [deque(maxlen=self.median_window_size_extended) for _ in range(0, l)]
        self.temp_measured = 0  # most recent measured temp
        self.temp_calculated = 0           # most recent calculated temp
        self.speed = 0          # most recent speed
        self.radial = None      # most recent radial angle. Invalid for room
        self.vertical = None    # most recent vertical angle. Invalid for room
        self.temp_med = 0
        self.speed_med = 0
        self.radial_med = None

        if self.algorithm is 0:
            self._prev_rel_phase = {}  # {(src, dst) : number}
            self._prev_abs_phase = {}  # {(src, dst) : number}
            self._cur_abs_phase = {}  # {(src, dst) : number}
        else:
            self._prev_abs_phase = {}
        self.start_calibration()

        # Added by Yannan
        self.past_5_counter = [0, 0, 0, 0, 0, 0]

        self.past_vx = []
        self.past_vy = []
        self.past_vz = []
        # duct is 4
        # End added by Yannan

    def get_distance(self):
        # d = 0.1875 if self.is_duct else 0.06
        if self.is_duct:
            return self.duct_distance
        else:
            return 0.06

    def process_reading(self, reading):
        if self.algorithm is 0:  # phase only
            self._process_reading_phase(reading)
        elif self.algorithm is 1:  # temperature guided
            self._process_reading_temperature(reading)
        else:
            print("Error: Invalid algorithm type")

    def _process_reading_phase(self, reading):
        # For each path, calculate absolute phase of reading 2 before max magnitude
        print(self.anemometer_id, "processing reading phase")
        num_sensors = reading.num_sensors
        timestamp = time.time() - self.start_time
        temp = reading.get_temperature()

        next_abs_phase, read_index = self._get_abs_phase(reading, self._cur_abs_phase)
        cur_rel_phase = self._infer_rel_phase(next_abs_phase)

        # If calibrating, track phase and max index
        if self.is_calibrating:
            for src in range(0, num_sensors):
                for dst in range(0, num_sensors):
                    if src == dst:
                        continue
                    if (src, dst) in cur_rel_phase:
                        self._calibration_phases[(src, dst)].append(cur_rel_phase[(src, dst)])
                    self._calibration_indices[(src, dst)].append(reading.get_max_index(src, dst))

            # finish calibration if applicable
            if len(self._calibration_phases) == num_sensors * (num_sensors - 1) and len(
                    self._calibration_phases[(0, 1)]) >= self.calibration_period:
                self.is_calibrating = False
                calibrated_phases, self._calibrated_index, _ = self._finish_calibration(self._calibration_phases,
                                                                                        self._calibration_indices, None)
                for (src, dst), phase in calibrated_phases.items():
                    cur_rel_phase[(src, dst)] -= phase

            # If include_calibration, also graph the phases during calibration period.
            if self.include_calibration and cur_rel_phase != {}:
                self._graph_calibration_phase(cur_rel_phase, self._cur_abs_phase, timestamp)

        # If not calibrating, calculate pairwise velocities.
        else:
            # TODO: scale distances properly for duct
            d = self.get_distance()
            speed = 0
            theta = None
            phi = None
            all_v_rel = []
            all_temps = []

            for i in range(len(self.paths)):
                (src, dst) = self.paths[i]
                phase_ab = cur_rel_phase[(src, dst)]
                phase_ba = cur_rel_phase[(dst, src)]
                abs_phase_ab = self._cur_abs_phase[(src, dst)]
                velocity, temp_calculated, success = self.phase_to_velocity_temp(phase_ab, phase_ba, d)
                if not success:
                    self._prev_rel_phase = cur_rel_phase
                    self._prev_abs_phase = self._cur_abs_phase
                    self._cur_abs_phase = next_abs_phase
                    return
                speed = velocity

                self.add_to_toggle_graph((timestamp, (phase_ab, phase_ba, abs_phase_ab, velocity)), i)
                if self.is_duct:
                    self.add_to_general_graph((timestamp, temp_calculated), i)
                all_v_rel.append(velocity)
                all_temps.append(temp_calculated)

            self._filter_velocity_outlier(all_v_rel, cur_rel_phase)

            # For room anemometer, also calculate vx, vy, vz, m, theta, phi. Weighted assuming node 1 at bottom.
            if not self.is_duct:
                vx, vy, vz = self.path_vel_to_directional_vel(all_v_rel)
                self._update_directional_vel(vx, vy, vz)
                speed, theta, phi = self.directional_velocities_to_spherical_coordinates(vx, vy, vz)
                vx_world, vy_world, vz_world = self.directional_velocities_to_world_coordinates(vx, vy, vz, reading)
                speed_world, theta_world, phi_world = self.directional_velocities_to_spherical_coordinates(vx_world, vy_world, vz_world)

                self.add_to_general_graph((timestamp, vx), 0)
                self.add_to_general_graph((timestamp, vy), 1)
                self.add_to_general_graph((timestamp, vz), 2)
                self.add_to_general_graph((timestamp, speed), 3)
                self.add_to_general_graph((timestamp, theta), 4, speed < self.low_velocity_threshold)
                self.add_to_general_graph((timestamp, phi), 5, speed < self.low_velocity_threshold)
                self.past_5_velocity_magnitudes.append(speed)

            if self.is_duct:
                self.add_to_strip_graph(timestamp, all_v_rel, all_temps)
                self.speed = np.mean(all_v_rel)
            else:
                self.add_to_strip_graph(timestamp, speed, all_temps, theta)
                self.speed = speed
            self.temp_measured = temp
            self.radial = theta
            self.vertical = phi
            self._graph_medians()

        # Rotate over the readings to prepare for the next reading.
        self._prev_rel_phase = cur_rel_phase
        self._prev_abs_phase = self._cur_abs_phase
        self._cur_abs_phase = next_abs_phase

    def _process_reading_temperature(self, reading):
        # Only append to buffers at the end of the function. Ideally, do locking to make buffers threadsafe
        print(self.anemometer_id, "processing reading using temperature")
        num_sensors = reading.num_sensors
        timestamp = time.time() - self.start_time
        temp = reading.get_temperature()

        abs_phases, read_indices = self._get_abs_phase(reading, self._prev_abs_phase)
        # Track index, phase, and temp during calibration
        if self.is_calibrating:
            for src in range(0, num_sensors):
                for dst in range(0, num_sensors):
                    if src == dst:
                        continue
                    self._calibration_indices[(src, dst)].append(reading.get_max_index(src, dst))
                    self._calibration_phases[(src, dst)].append(abs_phases[(src, dst)])
            self._calibration_temperatures.append(temp)

            # finish calibration if applicable
            if len(self._calibration_phases) == num_sensors * (num_sensors - 1) and len(
                    self._calibration_phases[(0, 1)]) >= self.calibration_period:
                self._calibrated_phase, self._calibrated_index, self._calibrated_temperature = self._finish_calibration(
                    self._calibration_phases, self._calibration_indices, self._calibration_temperatures)
                self._calibrated_TOF = self.temp_to_TOF(self._calibrated_temperature)

            if self.include_calibration:
                self._graph_calibration_phase(abs_phases, abs_phases, timestamp)
        # If not calibrating, calculate pairwise velocities
        else:
            d = self.get_distance()
            all_v_rel = []
            all_temps = []
            TOF_temp = self.temp_to_TOF(temp)
            TOF_diff = self._calibrated_TOF - TOF_temp
            f = 180000  # 180000 khz
            phase_diff = TOF_diff * f * 360
            speed = 0
            theta = None
            phi = None

            for i in range(len(self.paths)):
                (src, dst) = self.paths[i]
                corr_ab = self._phase_correction(src, dst, phase_diff, abs_phases)
                corr_ba = self._phase_correction(dst, src, phase_diff, abs_phases)
                avg = (corr_ab + corr_ba) / 2
                delta = (corr_ab - corr_ba) / 2
                phase_ab = phase_diff + avg + delta
                phase_ba = phase_diff + avg - delta
                abs_phase_ab = abs_phases[(src, dst)]
                velocity, temp_calculated, success = self.phase_to_velocity_temp(phase_ab, phase_ba, d)
                if not success:
                    self._prev_abs_phase = abs_phases
                    return
                speed = velocity

                self.add_to_toggle_graph((timestamp, (phase_ab, phase_ba, abs_phase_ab, velocity)), i)
                if self.is_duct:
                    self.add_to_general_graph((timestamp, temp_calculated), i)
                all_v_rel.append(velocity)
                all_temps.append(temp_calculated)

            # For room anemometer, also calculate vx, vy, vz, m, theta, phi. Weighted assuming node 1 at bottom.
            if not self.is_duct:
                vx, vy, vz = self.path_vel_to_directional_vel(all_v_rel)
                self._update_directional_vel(vx, vy, vz)
                speed, theta, phi = self.directional_velocities_to_spherical_coordinates(vx, vy, vz)
                vx_world, vy_world, vz_world = self.directional_velocities_to_world_coordinates(vx, vy, vz, reading)
                speed_world, theta_world, phi_world = self.directional_velocities_to_spherical_coordinates(vx_world,
                                                                                                           vy_world,
                                                                                                           vz_world)
                print("Sanity check local: ", speed, theta, phi)
                print("Sanity check world: ", speed_world, theta_world, phi_world)

                self.add_to_general_graph((timestamp, vx), 0)
                self.add_to_general_graph((timestamp, vy), 1)
                self.add_to_general_graph((timestamp, vz), 2)
                self.add_to_general_graph((timestamp, speed), 3)
                self.add_to_general_graph((timestamp, theta), 4, speed < self.low_velocity_threshold)
                self.add_to_general_graph((timestamp, phi), 5, speed < self.low_velocity_threshold)
                self.past_5_velocity_magnitudes.append(speed)

            if self.is_duct:
                self.add_to_strip_graph(timestamp, all_v_rel, all_temps)
                self.speed = np.mean(all_v_rel)
            else:
                self.add_to_strip_graph(timestamp, speed, all_temps, theta)
                self.speed = speed
            self.temp_measured = temp
            self.radial = theta
            self.vertical = phi
            self._graph_medians()

        # Save absolute phase for reference
        self._prev_abs_phase = abs_phases

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
                magnitude_per_wave[path_string] = (path_reading.mag[index + 1] - path_reading.mag[index]) / 8

            if path_string == "0_to_1":
                print("using index ", index, "from mags", path_reading.mag, ", got ", magnitude[path_string])

        # If calibrating, track phase and magnitude
        if self.is_calibrating:
            for path_string, p in abs_phase.items():
                self._calibration_phases[path_string].append(p)
            for path_string, m in magnitude.items():
                self._calibration_magnitudes[path_string].append(m)

            # finish calibration if applicable
            if len(self._calibration_phases) == 16 and len(
                    self._calibration_phases["0_to_1"]) >= self.calibration_period:
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

                for i in range(-max_wave_change, max_wave_change + 1):
                    v = m_c + ((p - p_c) / 360 + i) * mpw  # try + and - ...
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
                d = self.get_distance()
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
                    v_rel = -v_rel  # sign is flipped for four path for other anemometer
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
                vx_weight, vy_weight, vz_weight = self.get_directional_velocity_weights
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
            self._graph_medians()

    # ================PUBLIC FUNCTIONS=================
    def start_calibration(self):
        self.is_calibrating = True
        if self.algorithm is 0:
            self._calibration_phases = defaultdict(
                list)  # {(src, dst) : []}, used to hold rel phases during calibration period
            self._calibration_indices = defaultdict(
                list)  # {src, dst): []}, used to hold indices of max-2 during calibration period
            self._calibrated_index = {}  # {src, dst): index}
        else:
            self._calibration_indices = defaultdict(list)
            self._calibration_phases = defaultdict(list)
            self._calibration_temperatures = []

            self._calibrated_index = {}
            self._calibrated_phase = {}
            self._calibrated_temperature = 0
            self._calibrated_TOF = 0

    def generate_window(self):
        self.aw = ApplicationWindow(None, self, self.is_duct, self.anemometer_id, self.paths)
        return self.aw

    def dump_raw_data(self):
        self.data_dump_func()

    def update_medians(self, median_window_size, graph_type, index):
        # Todo: small bug where buffer may still hold a few medians using the old median window size.
        x_medians = []
        y_medians = []
        if graph_type is "toggle":
            if len(self.relative_data[index]) < median_window_size:
                return [], []
            for i in range(len(self.relative_data[index]) - median_window_size + 1):
                x_medians.append(np.median([x[0] for x in self.relative_data[index][i: i + median_window_size]]))
                y_medians.append(np.median([x[1][3] for x in self.relative_data[index][i: i + median_window_size]]))
        elif graph_type is "general":
            if len(self.general_data[index]) < median_window_size:
                return [], []
            for i in range(len(self.general_data[index]) - median_window_size + 1):
                x_medians.append(np.median([x[0] for x in self.general_data[index][i: i + median_window_size]]))
                y_medians.append(np.median([x[1] for x in self.general_data[index][i: i + median_window_size]]))
        elif graph_type is "strip":
            if len(self.strip_data[index]) < median_window_size:
                return [], []
            for i in range(len(self.strip_data[index]) - median_window_size + 1):
                x_medians.append(np.median([x[0] for x in self.strip_data[index][i: i + median_window_size]]))
                if not self.is_duct:
                    y_medians.append(np.median([x[1] for x in self.strip_data[index][i: i + median_window_size]]))
                else:
                    y_medians = [[] for _ in range(len(self.paths))]
                    for path in range(len(self.paths)):
                        y_medians[path].append(np.median([x[1][path] for x in self.strip_data[index][i: i + median_window_size]]))
        return x_medians, y_medians

    def temp_to_TOF(self, temperature):
        speed = 331.5 + 0.607 * temperature  # speed of sound in m/s, where temperature is in C
        return self.get_distance() / speed

    def velocity_to_temp(self, velocity):
        # calibrated black magic
        return (-(velocity - 331.5)/0.607+18.5)*2+24

    def phase_to_velocity_temp(self, phase_ab, phase_ba, dist):
        tof_ab = phase_ab / (360 * 180000)  # khz?
        tof_ba = phase_ba / (360 * 180000)

        if self.is_duct:
            v_ab = dist / (dist / 344 - tof_ab)
            v_ba = dist / (dist / 344 - tof_ba)
        else:
            v_ab = dist / (dist / 343 + tof_ab)
            v_ba = dist / (dist / 343 + tof_ba)

        v_rel = -1 * (v_ab - v_ba) / 2 / np.cos(45 * np.pi / 180.)
        avg_v = (v_ab + v_ba) / 2
        if abs(v_rel) >= 6:
            print("v_rel > 6", phase_ab, phase_ba)
            return 0, 0, False     # do blank instead
        # temp = + avg_v * avg_v / 400 - 273.15
        temp = self.velocity_to_temp(avg_v)
        return v_rel, temp, True

    # Assumes node 1 at bottom.
    def get_directional_velocity_weights(self, w=None, v=None):
        sin30 = np.sin(np.pi/180 * 30)
        sin60 = np.sin(np.pi/180 * 60)
        cos_bot = np.cos(np.pi/180 * 54.74)  # Angle between top plane and bottom transducer is 54.7356 degrees, not 60
        sin_bot = np.sin(np.pi/180 * 54.74)

        vx_weight = [sin30 * cos_bot, sin60, 0, cos_bot, -sin30 * cos_bot, -sin60]
        vy_weight = [sin60 * cos_bot, sin30, 1, 0, sin60 * cos_bot, sin30]
        vz_weight = [-sin_bot, 0, 0, sin_bot, sin_bot, 0]
        # vx_weight = [sin30 * sin30, sin60, 0, sin30, -sin30 * sin30, -sin60]
        # vy_weight = [sin60 * sin30, sin30, 1, 0, sin60 * sin30, sin30]
        # vz_weight = [-sin60, 0, 0, sin60, sin60, 0]
        if w is not None:
            vx_weight = [0, sin60 * w, 0, 0, 0, -sin60 * w]
            vy_weight = [0, sin30 * w, 1 * w, 0, 0, sin30 * w]
        if v is not None:
            vx_weight = [sin30 * cos_bot * v, 0, 0, cos_bot * v, -sin30 * cos_bot * v, 0]
            vy_weight = [sin60 * cos_bot * v, 0, 0, 0, sin60 * cos_bot * v, 0]
        return vx_weight, vy_weight, vz_weight

    # Calculate directional velocities (vx, vy, vz) for room anemometer.
    # Weighted assuming node 1 at bottom. Path velocities must be in order
    def path_vel_to_directional_vel(self, path_vel):
        # path_vel = self._filter_path_vel(path_vel)
        if self.use_room_min:
            w = 0.85
            v = 1.1
        else:
            w = 0.9
            v = 1.15    # Todo: these are placeholders
        vx_weight, vy_weight, vz_weight = self.get_directional_velocity_weights()
        reweighted_w = False
        tnx = 5  # 6 if 15degree coordinate system
        tny = 5  # 6 if 15degree coordinate system
        tnz = []

        # Ali's heuristic-based re-weighting
        if 0 > abs(path_vel[1]) / path_vel[2] > -0.5 and 0 > abs(path_vel[1]) / path_vel[5] > -0.5:
            reweighted_w = True
            tnz = [0, 3]
        if 0 > abs(path_vel[2]) / path_vel[1] > -0.5 and 0 < abs(path_vel[2]) / path_vel[5] < 0.5:
            reweighted_w = True
            tnz = [0, 4]
        if 0 < abs(path_vel[5]) / path_vel[1] < 0.5 and 0 < abs(path_vel[5]) / path_vel[2] < 0.5:
            reweighted_w = True
            tnz = [3, 4]
        if reweighted_w:
            vx_weight, vy_weight, vz_weight = self.get_directional_velocity_weights(w=w)
            tnx = 2
            tny = 3

        if (0 < abs(path_vel[0] / path_vel[3]) < 0.5 and 0 < abs(path_vel[0] / path_vel[4]) < 0.5) or \
                (0 < abs(path_vel[3] / path_vel[0]) < 0.5 and 0 < abs(path_vel[3] / path_vel[4]) < 0.5) or \
                (0 < abs(path_vel[4] / path_vel[0]) < 0.5 and 0 < abs(path_vel[4] / path_vel[0]) < 0.5):
            vx_weight, vy_weight, vz_weight = self.get_directional_velocity_weights(v=v)
            tnx = 3
            tny = 2

        vx = sum(sorted([i[0] / i[1] for i in zip(path_vel, vx_weight) if i[1] != 0])) / tnx
        vy = sum(sorted([i[0] / i[1] for i in zip(path_vel, vy_weight) if i[1] != 0])) / tny
        vz = sum(sorted([i[0] / i[1] for i in zip(path_vel, vz_weight) if i[1] != 0])) / 3
        if len(tnz) > 1:
            vp = np.sqrt(vx * vx + vy * vy)
            # vz = (all_v_rel[tnz[0]]/vz_weight[tnz[0]]+all_v_rel[tnz[1]]/vz_weight[tnz[1]])/2
        # vx = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vx_weight) if i[1] != 0])) / 5
        # vy = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vy_weight) if i[1] != 0])) / 5
        # vz = sum(sorted([i[0] / i[1] for i in zip(all_v_rel, vz_weight) if i[1] != 0])) / 3
        return vx, vy, vz

    def directional_velocities_to_spherical_coordinates(self, vx, vy, vz):
        # theta calculation
        avg_m = 0 if len(self.past_5_velocity_magnitudes) == 0 else sum(self.past_5_velocity_magnitudes) / len(
            self.past_5_velocity_magnitudes)
        theta = np.arctan2(vy, vx) * 180 / np.pi if avg_m > 0.5 else 0

        # m calculation
        m = np.sqrt(vx * vx + vy * vy + vz * vz)
        # Simple low speed positive-bias filter: linearly interpolate towards 0 when speed below low_velocity_threshold
        m = self._filter_magnitude(m)

        # phi calculation based on
        # Based on past 10 vx, vy, vz, recalculate m to use in phi calculations when sqrt(vx^2 + vy^2) < 0.5
        # to cancel out noise and approach 0.
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
                phi = np.arcsin(vz / m) * 180 / np.pi if avg_m > 0.5 else 0
        return m, theta, phi

    def directional_velocities_to_world_coordinates(self, vx, vy, vz, reading):
        # v_anem = world_to_anem * v_world
        # world_to_anem_inverse * v_anem = v_world
        v_anem = [vx, vy, vz]
        world_to_anem = reading.get_rotation_matrix()
        try:
            anem_to_world = np.linalg.inv(world_to_anem)
            v_world = np.dot(anem_to_world, v_anem)
            return v_world[0, 0], v_world[0, 1], v_world[0, 2]
        except np.linalg.linalg.LinAlgError:
            print("Warning: couldn't invert anemometer rotation matrix.")
            return 0, 0, 0


    def add_to_general_graph(self, point, index, is_blank=False):
        if not is_blank:
            self.general_graph_buffer[index].append(point)
            self.general_data[index].append(point)
        else:
            self.general_graph_buffer_blank[index].append((point[0], 0))    # graph this point at (x, 0) and gray it out

    def add_to_toggle_graph(self, point, index):
        self.toggle_graph_buffer[index].append(point)
        self.relative_data[index].append(point)

    def add_to_strip_graph(self, timestamp, speeds, temps, radial=None):
        if not self.is_duct:
            temps = np.mean(temps)  # Graph the average of the path temperatures for room anemometer
        self.strip_graph_buffer[0].append((timestamp, speeds))
        self.strip_data[0].append((timestamp, speeds))
        self.strip_graph_buffer[1].append((timestamp, temps))
        self.strip_data[1].append((timestamp, temps))
        if radial is not None:
            speed = np.mean(speeds)
            if speed < self.low_velocity_threshold:
                self.strip_graph_buffer_blank[2].append((timestamp, 0))
            else:
                self.strip_graph_buffer[2].append((timestamp, radial))
                self.strip_data[2].append((timestamp, radial))

    def get_speed(self):
        return self.speed_med

    def get_temp_measured(self):
        return self.temp_measured

    def get_radial(self):
        return self.radial_med

    # todo: save vertical medians too
    def get_vertical(self):
        return self.vertical

    def get_averaging_window(self):
        return self.median_window_size

    # ================HELPER FUNCTIONS=================

    # Returns map of (src, dst) to absolute phase from reading, as well as read index.
    def _get_abs_phase(self, reading, default_phase=None):
        num_sensors = reading.num_sensors
        data_len = 4
        abs_phases = {}  # {(src, dst) : phase in degrees}
        read_indices = {}  # {(src, dst) : index offset from this reading's max on which to do calculations}
        for src in range(0, num_sensors):
            for dst in range(0, num_sensors):
                if src == dst:
                    continue

                if self.is_calibrating:
                    offset_from_max = -2
                else:
                    calibrated_max_index = self._calibrated_index[(src, dst)]
                    cur_max_index = reading.get_max_index(src, dst)
                    offset_from_max = -2 + (calibrated_max_index - cur_max_index)
                read_index = data_len + offset_from_max - 1
                read_indices[(src, dst)] = read_index

                if read_index < 0 or read_index >= data_len:
                    print(self.anemometer_id, "ERROR: Couldn't find maximum magnitude for path ", (src, dst),
                          ", index out of range.\n\t", "current max index:", cur_max_index,
                          "calibrated max index:", self._calibrated_index[(src, dst)],
                          "read index:", read_index)
                    if default_phase is not None and (src, dst) in default_phase:
                        phase = default_phase[(src, dst)]
                        print("Replacing with phase ", phase)
                        abs_phases[(src, dst)] = phase
                    else:
                        print("Replacing with phase 0.")
                        abs_phases[(src, dst)] = 0
                else:
                    i = reading.get_imaginary(src, dst)[read_index]
                    q = reading.get_real(src, dst)[read_index]
                    abs_phases[(src, dst)] = np.arctan2(i, q) * 180 / np.pi
        return abs_phases, read_indices

    # Infer current reading's relative phase, using previous relative phase, and previous, current,
    # and next absolute phases. See README
    def _infer_rel_phase(self, next_abs_phase):
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
                next_wrapped_delta = next_delta if abs(next_delta) < 180 else -1 * next_sign * (
                        360 - abs(next_delta))

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
        return cur_rel_phase

    def _graph_calibration_phase(self, relative_phases, absolute_phases, timestamp):
        for i in range(len(self.paths)):
            (src, dst) = self.paths[i]
            phase_ab = relative_phases[(src, dst)]
            phase_ba = relative_phases[(dst, src)]
            abs_phase_ab = absolute_phases[(src, dst)]
            self.toggle_graph_buffer[i].append(timestamp, (phase_ab, phase_ba, abs_phase_ab, 0))

    def _graph_medians(self):
        if len(self.general_data[0]) >= self.median_window_size:
            # Find medians for general and toggle graphs
            for i in range(len(self.paths)):
                y_med = np.median([x[1] for x in self.general_data[i][-self.median_window_size:]])
                x_med = np.median([x[0] for x in self.general_data[i][-self.median_window_size:]])
                self.general_graph_buffer_med[i].append((x_med, y_med))

                y_med = np.median(
                    [x[1][3] for x in self.relative_data[i][-self.median_window_size:]])  # Median relative velocity
                x_med = np.median([x[0] for x in self.relative_data[i][-self.median_window_size:]])
                self.toggle_graph_buffer_med[i].append((x_med, y_med))

            # Find medians for strip graphs
            # speed
            if not self.is_duct:
                y_med = np.median([x[1] for x in self.strip_data[0][-self.median_window_size:]])
                x_med = np.median([x[0] for x in self.strip_data[0][-self.median_window_size:]])
                self.strip_graph_buffer_med[0].append((x_med, y_med))
                self.speed_med = y_med
            else:
                y_meds = []
                x_med = np.median([x[0] for x in self.strip_data[0][-self.median_window_size:]])
                for i in range(len(self.paths)):
                    y_med = np.median([x[1][i] for x in self.strip_data[0][-self.median_window_size:]])
                    y_meds.append(y_med)
                self.strip_graph_buffer_med[0].append((x_med, y_meds))
                self.speed_med = np.mean(y_meds)
            # temp
            if not self.is_duct:
                y_med = np.median([x[1] for x in self.strip_data[1][-self.median_window_size:]])
                x_med = np.median([x[0] for x in self.strip_data[1][-self.median_window_size:]])
                self.strip_graph_buffer_med[1].append((x_med, y_med))
            else:
                y_meds = []
                x_med = np.median([x[0] for x in self.strip_data[1][-self.median_window_size:]])
                for i in range(len(self.paths)):
                    y_med = np.median([x[1][i] for x in self.strip_data[1][-self.median_window_size:]])
                    y_meds.append(y_med)
                self.strip_graph_buffer_med[1].append((x_med, y_meds))
            # azimuth
            if not self.is_duct:
                y_med = np.median([x[1] for x in self.strip_data[2][-self.median_window_size:]])
                x_med = np.median([x[0] for x in self.strip_data[2][-self.median_window_size:]])
                self.strip_graph_buffer_med[2].append((x_med, y_med))
                self.radial_med = y_med

    def _median_in_window(self, data, window_size=None):
        if window_size is None:
            window_size = self.median_window_size
        if len(data) == 0:
            print("Warn: taking median of empty list. Returning 0")
            return 0
        return np.median(data[-window_size:])

    def _finish_calibration(self, calibration_phases=None, calibration_indices=None, calibration_temperatures=None):
        self.is_calibrating = False
        calibrated_phases = {}
        calibrated_indices = {}
        calibrated_temp = 0

        if calibration_phases is not None:
            for (src, dst), phase_list in calibration_phases.items():
                # make the phase list continuous. Ex: 170, -150, -130 -> 170, 210, 230
                for i in range(1, len(phase_list)):
                    d = closest_rotation(phase_list[i - 1], phase_list[i])
                    phase_list[i] = phase_list[i - 1] + d

                print("I am ", self.anemometer_id, " with path ", (src, dst), ", phases: ", phase_list)
                # calibrated_phases[(src, dst)] = mean_within_sd(phase_list, 2)
                calibrated_phases[(src, dst)] = np.median(phase_list)

        if calibration_indices is not None:
            for (src, dst), index_list in calibration_indices.items():
                index_list = [i for i in index_list if i >= 0]  # ignore all invalid indices
                index_mode = 0
                if len(index_list) > 0:
                    index_mode = max(set(index_list), key=index_list.count)
                calibrated_indices[(src, dst)] = index_mode
                print(self.anemometer_id, "index[", (src, dst), "] = ", index_list, ", mode =", index_mode)

        if calibration_temperatures is not None:
            # calibrated_temp = mean_within_sd(calibration_temperatures, 2)
            calibrated_temp = np.median(calibration_temperatures)

        return calibrated_phases, calibrated_indices, calibrated_temp

    # Begin added by Yannan
    # place check after for loop, all values have been appended
    # self.toggle_graph_buffer[i].appends v_rel as well --> toggle canvas issue, second priority?
    # otherwise, also need to pass this in and edit somehow (with temporary list, then extend + append possibly)
    def _filter_velocity_outlier(self, all_v_rel, cur_rel_phase):
        outlier_index = -1
        for i in range(len(all_v_rel)):
            # Take mean of all values except value in question (not averaging whole thing?)
            rest_of_lst = all_v_rel[0:i] + all_v_rel[i + 1:]
            avg = mean([abs(val) for val in rest_of_lst])
            condition = avg < 0.5 and abs(all_v_rel[i]) > (avg + 1)
            if condition:
                outlier_index = i
                continue  # assumes only one outlier
        if outlier_index != -1:
            # counter for each index 
            self.past_5_counter[outlier_index] += 1
            # > or >=?
            if self.past_5_counter[outlier_index] >= 20:
                # change cur_rel_phase, only change for incorrect index; mean of the rest of indices
                (src, dst) = self.paths[outlier_index]
                (dst, src) = self.paths[outlier_index]
                new_val = (sum(cur_rel_phase.values()) - cur_rel_phase[(src, dst)] - cur_rel_phase[(dst, src)]) / (
                        len(cur_rel_phase) - 2)
                cur_rel_phase[(src, dst)] = new_val
                cur_rel_phase[(dst, src)] = new_val
                # all_v_rel[outlier_index] = avg
                self.past_5_counter[outlier_index] = 0
        else:
            # reset counter if condition is not seen? intermittent issues?
            for count in self.past_5_counter:
                count = 0

    def _filter_path_vel(self, path_vel):
        filtered = list(path_vel)
        for i in range(len(path_vel)):
            self.path_vels[i].append(path_vel[i])
            if abs(path_vel[i]) < 0.5:
                med = np.median(self.path_vels[i])
                if abs(med) < abs(path_vel[i]):
                    # print("lower pathvel: ", med, path_vel[i])
                    filtered[i] = med
        return filtered

    def _filter_magnitude(self, m):
        if m < self.low_velocity_threshold:
            if m <= self.m_zero:
                m = 0
            else:
                m = (m - self.m_zero)/(self.low_velocity_threshold - self.m_zero) * self.low_velocity_threshold
        return m
        # OTHER m FILTERING METHODS. Not used.
        # m calculation based on means in low windspeed
        # if abs(vx) < 0.5 and abs(vy) < 0.5 and abs(vz) < 0.5:
        #     temp_vx = mean(self.past_vx)
        #     temp_vy = mean(self.past_vy)
        #     temp_vz = mean(self.past_vz)
        #     m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(temp_vz, 2))

        # # m calculation based on medians
        # temp_vx = self._median_in_window(self.past_vx)
        # temp_vy = self._median_in_window(self.past_vy)
        # temp_vz = self._median_in_window(self.past_vz)
        # m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(temp_vz, 2))
        #
        # # m re-calculation for low windspeeds (to reduce positive bias in zero-wind situations)
        # if m < 0.5:
        #     temp_vx = self._median_in_window(self.past_vx, self.median_window_size_extended)
        #     temp_vy = self._median_in_window(self.past_vy, self.median_window_size_extended)
        #     temp_vz = self._median_in_window(self.past_vz, self.median_window_size_extended)
        #     temp_m = np.sqrt(pow(temp_vx, 2) + pow(temp_vy, 2) + pow(temp_vz, 2))
        #     m = min(m, temp_m)

    def _update_directional_vel(self, vx, vy, vz):
        if len(self.past_vx) < 10:
            self.past_vx.append(vx)
            self.past_vy.append(vy)
            self.past_vz.append(vz)
        else:
            self.past_vx = self.past_vx[1:] + [vx]
            self.past_vy = self.past_vy[1:] + [vy]
            self.past_vz = self.past_vz[1:] + [vz]

    def _phase_correction(self, src, dst, expected_phase_difference, absolute_phases):
        expected_phase = self._calibrated_phase[(src, dst)] + expected_phase_difference
        actual_phase = absolute_phases[(src, dst)]
        phase_correction = closest_rotation(expected_phase, actual_phase)
        # print("Path ", (src, dst), ", calibrated ", self._calibrated_phase[(src, dst)], ", expected change ",
        #       expected_phase_difference, ", actual ", absolute_phases[(src, dst)], " correction: ", phase_correction)
        return phase_correction


# Returns the rotation from x to y, such that x + rotation = y
# abs(rotation) < 180, and y is truncated to be between 0 and 360
# ex: closest_rotation(40, -40) = -80. closest_rotation(420, 190) = 130. closest_rotation(420, -530) = 130
def closest_rotation(x, y):
    d = (y - x) % 360
    if d > 180:
        d -= 360
    return d

    # Completes calibration and returns the calibrated phase and index


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def mean_within_sd(nums, sd_boundary):
    m = np.mean(nums)
    sd = np.std(nums)
    if sd == 0:
        return m
    s = 0
    count = 0
    for num in nums:
        if abs(num - m) / sd <= sd_boundary:
            s += num
            count += 1
    if count == 0:
        # What should we actually do in this case?
        print("Warn: Given list is very noisy. Could not get clean average")
        return m
    else:
        return s / count


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
    #             vx_weight, vy_weight, vz_weight = self.get_directional_velocity_weights()
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
