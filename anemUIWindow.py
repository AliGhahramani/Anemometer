# Holds classes for starting up a UI window for tracking anemometer readings

from __future__ import unicode_literals
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import time, os, threading
import numpy as np

GRAPH_MAX_X_WINDOW = 200


class MultipleApplicationWindows(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.__windows = []

    def add_window(self, window):
        self.__windows.append(window)

    def show(self):
        for window in self.__windows:
            window.show()  # probably show will do the same trick


class ApplicationWindow(QtWidgets.QDialog):
    is_duct = True
    anemometer_id = ""

    def __init__(self, parent, anem_processor_owner, is_duct, anemometer_id, paths):
        QtWidgets.QDialog.__init__(self, parent)
        self.anem_processor_owner = anem_processor_owner  # owning AnemometerProcessor object from which to fetch stream
        self.is_duct = is_duct
        self.anemometer_id = anemometer_id
        self.paths = paths

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Anemometer " + str(anemometer_id))
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.main_widget = QtWidgets.QWidget(self)

        # Set up tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()

        self.tabs.addTab(self.tab1, "Main")
        self.tabs.addTab(self.tab2, "Diagnostic")
        self.set_up_main_tab()
        self.set_up_diagnostic_tab()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Set up settings bar
        self.median_window_textbox = QtWidgets.QLineEdit(self)
        self.median_window_textbox.setText('5')  # default median window size is 5
        self.median_window_textbox.setMaxLength(2)
        self.median_window_label = QtWidgets.QLabel('Median window size', self)
        self.median_window_button = QtWidgets.QPushButton('Set', self)
        self.median_window_button.clicked.connect(self.on_set_median_window_click)

        settings_layout = QtWidgets.QHBoxLayout(self)
        settings_layout.addWidget(self.median_window_label)
        settings_layout.addWidget(self.median_window_textbox)
        settings_layout.addWidget(self.median_window_button)
        settings_layout.addStretch(1)
        layout.addLayout(settings_layout)

        self.main_widget.setFocus()
        # self.setCentralWidget(self.main_widget)
        # if self.anem_processor_owner.is_calibrating:
        #     self.statusBar().showMessage("CALIBRATING. Please wait")

        print("CREATED WINDOW " + self.anemometer_id)

    def set_up_main_tab(self):
        # Create main tab layout
        # SPEED     azimuth     vertical        TEMP        averaging window
        # Speed, azimuth, temp strip charts
        # Units: [toggle] m/s, fpm, mph, C, F. interval: [toggle] 1, 2, 3, 4, 5

        # SPEED TEMP avg window
        # Speed, temp strip charts
        # Units: [toggle] m/s, fpm, mph, C, F. interval: [toggle] 1, 2, 3, 4, 5

        # Set up header with speed, azimuth, vertical, temp, and averaging window
        l1 = QtWidgets.QVBoxLayout(self.tab1)
        self.speed_label = QtWidgets.QLabel('0.0 m/s')
        self.small_font = self.speed_label.font()
        self.big_font = self.speed_label.font()
        self.big_font.setPointSize(self.small_font.pointSize() + 7)
        self.speed_label.setFont(self.big_font)
        self.temp_label = QtWidgets.QLabel('0.0 °C')
        self.temp_label.setFont(self.big_font)
        self.window_label = QtWidgets.QLabel('')
        header_l = QtWidgets.QHBoxLayout()
        header_l.addWidget(self.speed_label)
        header_l.addWidget(self.temp_label)
        if not self.is_duct:
            self.azimuth_label = QtWidgets.QLabel('azimuth: 0 °')
            self.vertical_label = QtWidgets.QLabel('vertical: 0 °')
            header_l.addWidget(self.azimuth_label)
            header_l.addWidget(self.vertical_label)
        header_l.addWidget(self.window_label)
        l1.addLayout(header_l)

        # Create thread to update header periodically
        t = threading.Thread(target=self._update_main_tab_header)
        t.daemon = True
        t.start()

        # Set up strip graphs
        strip_l = QtWidgets.QHBoxLayout()
        # TODO: maybe set parent to this tab, instead of main_widget?
        self.strip_graphs = StripGraphs(self, self.anem_processor_owner, self.is_duct, self.main_widget, width=5,
                                        height=4, dpi=100)
        strip_l.addWidget(self.strip_graphs)
        l1.addLayout(strip_l)

        # Set up settings bar
        settings_l = QtWidgets.QHBoxLayout()
        self.units_toggle = QtWidgets.QLabel("units toggle here")
        self.interval_toggle = QtWidgets.QLabel("interval toggle here")
        l1.addLayout(settings_l)

    def set_up_diagnostic_tab(self):
        # Create diagnostic tab layout
        l2 = QtWidgets.QGridLayout(self.tab2)
        # Add buttons
        self.debug_toggle_button = QtWidgets.QPushButton('TOGGLE DEBUG', self)
        self.debug_toggle_button.setToolTip('toggle between normal mode (velocities) and debug mode (relative phases)')
        self.debug_toggle_button.clicked.connect(self.on_toggle_click)
        self.is_debug = False
        self.dump_data_button = QtWidgets.QPushButton('SAVE RAW DATA', self)
        self.dump_data_button.setToolTip('save all raw data to anemometer_raw_data.txt')
        self.dump_data_button.clicked.connect(self.on_dump_raw_data_click)
        self.dump_graph_button = QtWidgets.QPushButton('SAVE GRAPH DATA', self)
        self.dump_graph_button.setToolTip('save all data shown on graph to anemometer_graph_data_' +
                                          str(self.anemometer_id) + '.tsv')
        self.dump_graph_button.clicked.connect(self.on_dump_graph_data_click)
        l2.addWidget(self.debug_toggle_button, 0, 0)
        l2.addWidget(self.dump_data_button, 1, 0)
        l2.addWidget(self.dump_graph_button, 2, 0)

        # Add graphs
        # TODO: Probably do this with subplots instead of multiple plots..
        self.toggle_graphs = []
        self.general_graphs = []
        for i in range(len(self.paths)):
            g = ToggleableGraph(self, self.anem_processor_owner, self.paths, i, self.main_widget, width=5, height=4,
                                dpi=100)
            self.toggle_graphs.append(g)
            # l2.addWidget(g)
            l2.addWidget(g, i % 2, i // 2 + 1)
        if self.is_duct:
            # Graphs for temperature
            for i in range(len(self.paths)):
                path_str = str(self.paths[i][0]) + " to " + str(self.paths[i][1])
                g = GeneralGraph(self, self.anem_processor_owner, i, self.main_widget, width=5, height=4, dpi=100,
                                 title="Temp on path " + path_str, ylabel="temperature (C)", yrange=(21, 25))
                self.general_graphs.append(g)
                l2.addWidget(g, 2 + i % 2, i // 2 + 1)
        else:
            # Graphs for vx, vy, vz, m, theta, phi
            vx = GeneralGraph(self, self.anem_processor_owner, 0, self.main_widget, width=5, height=4, dpi=100)
            vy = GeneralGraph(self, self.anem_processor_owner, 1, self.main_widget, width=5, height=4, dpi=100)
            vz = GeneralGraph(self, self.anem_processor_owner, 2, self.main_widget, width=5, height=4, dpi=100)
            m = GeneralGraph(self, self.anem_processor_owner, 3, self.main_widget, width=5, height=4, dpi=100,
                             title="Wind speed", ylabel="speed (m/s)")
            theta = GeneralGraph(self, self.anem_processor_owner, 4, self.main_widget, width=5, height=4, dpi=100,
                                 title="Radial angle", ylabel="radial angle (degrees)", yrange=(-220, 220),
                                 yticks=[-180, -90, 0, 90, 180])
            phi = GeneralGraph(self, self.anem_processor_owner, 5, self.main_widget, width=5, height=4, dpi=100,
                               title="Vertical angle", ylabel="vertical angle (degrees)", yrange=(-100, 100),
                               yticks=[-90, -60, -30, 0, 30, 60, 90])
            self.general_graphs.extend([vx, vy, vz, m, theta, phi])
            l2.addWidget(vx, 2, 1)
            l2.addWidget(vy, 2, 2)
            l2.addWidget(vz, 2, 3)
            l2.addWidget(m, 3, 1)
            l2.addWidget(theta, 3, 2)
            l2.addWidget(phi, 3, 3)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def on_toggle_click(self):
        for graph in self.toggle_graphs:
            graph.toggle()

    def on_dump_raw_data_click(self):
        self.anem_processor_owner.dump_raw_data()

    def on_set_median_window_click(self):
        try:
            new_median_window_size = int(self.median_window_textbox.text())
        except ValueError:
            print("Error: Cannot use a non-integer median window. You entered ", self.median_window_textbox.text())
            return
        if self.anem_processor_owner.median_window_size != new_median_window_size:
            self.anem_processor_owner.median_window_size = new_median_window_size
            for graph in self.toggle_graphs:
                graph.update_median(new_median_window_size)
            for graph in self.general_graphs:
                graph.update_median(new_median_window_size)
            self.strip_graphs.update_median(new_median_window_size)

    def on_dump_graph_data_click(self):
        filename = 'anemometer_graph_data_' + str(self.anemometer_id) + '.tsv'
        print("Attempting to save data to " + filename)
        f = open(resource_path(filename), 'w')

        # write column labels
        f.write("time since start")
        for graph in self.toggle_graphs:
            f.write("\t" + str(graph.data_label()))
        for graph in self.general_graphs:
            f.write("\t" + str(graph.data_label()))
        f.write("\n")

        # write all data lines
        for i in range(len(self.toggle_graphs[0].xdata)):
            f.write(str(self.toggle_graphs[0].xdata[i]))  # time since start
            for graph in self.toggle_graphs:
                if i < len(graph.ydata_vel):
                    f.write("\t" + str(graph.ydata_vel[i]))
            for graph in self.general_graphs:
                if i < len(graph.ydata):
                    f.write("\t" + str(graph.ydata[i]))
            f.write("\n")

        f.close()
        print("Successfully saved data to " + filename)

    def _update_main_tab_header(self):
        while True:
            speed = self.anem_processor_owner.get_speed()
            temp = self.anem_processor_owner.get_temp_measured()
            radial = self.anem_processor_owner.get_radial()
            vertical = self.anem_processor_owner.get_vertical()
            window = self.anem_processor_owner.get_averaging_window()
            self.set_main_tab_header(speed, temp, window, radial, vertical)
            time.sleep(1)

    def set_main_tab_header(self, speed, temp, averaging_window, azimuth=None, vertical=None):
        self.speed_label.setText(strformat_double(speed) + " m/s")
        self.temp_label.setText(strformat_double(temp) + " °C")
        self.window_label.setText("averaging over " + str(averaging_window) + " readings")

        if azimuth is not None and not self.is_duct:
            self.azimuth_label.setText("azimuth: " + strformat_double(azimuth) + " °")
        if vertical is not None and not self.is_duct:
            self.vertical_label.setText("vertical: " + strformat_double(vertical) + " °")


class StripGraphs(FigureCanvas):
    # Initialize a set of strip graphs using subplots that pull from strip_graph_buffer
    # Note: this is pretty sketchy OOP-wise; probably should have done it with 3 separate graphs instead of subplots
    def __init__(self, app_window, anem_processor_owner, is_duct, parent=None,
                 width=5, height=4, dpi=100):
        self.aw = app_window
        self.anem_processor_owner = anem_processor_owner
        self.is_duct = is_duct
        self.include_radial = not is_duct  # include azimuth strip graph for room anemometer
        self.start_time = time.time()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.num_graphs = 3 if self.include_radial else 2

        self.title = "title placeholder"
        self.scale_y_axes = [True, True]
        self.axes = []
        self.axes.append(self.fig.add_subplot(self.num_graphs, 1, 1))
        self.axes.append(self.fig.add_subplot(self.num_graphs, 1, 2))
        self.axes[0].set_ylabel("velocity (m/s)")
        self.axes[1].set_ylabel("temp (°C)")
        if self.include_radial:
            self.axes.append(self.fig.add_subplot(self.num_graphs, 1, 3))
            self.axes[2].set_ylabel("azimuth (°)")
            self.scale_y_axes.append(False)
        for axis in self.axes:
            axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            axis.grid(color='gray', linewidth=0.5, which='major', linestyle='-')
            axis.grid(color='lightgray', linewidth=0.5, which='minor', linestyle='--')
        self.axes[-1].set_xlabel("time since start (s)")
        self.fig.set_tight_layout(True)

        self.xdata = [[] for _ in range(self.num_graphs)]
        self.ydata = [[] for _ in range(self.num_graphs)]
        self.xdata_med = [[] for _ in range(self.num_graphs)]
        self.ydata_med = [[] for _ in range(self.num_graphs)]
        self.ymin = [0 for _ in range(self.num_graphs)]
        self.ymax = [0 for _ in range(self.num_graphs)]
        self.ymin_index = [(0, 0) for _ in range(self.num_graphs)]     # each tuple is (x value, index at min y point)
        self.ymax_index = [(0, 0) for _ in range(self.num_graphs)]
        self.ln = [None for _ in range(self.num_graphs)]
        self.ln_med = [None for _ in range(self.num_graphs)]

        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.animation = self.animate()

    def data_label(self):
        return self.title

    def compute_initial_figure(self):
        for i in range(self.num_graphs):
            if not self.is_duct:
                # 2 lines per graph: one for raw data, one for medians
                ln, = self.axes[i].plot(self.xdata[i], self.ydata[i], '.', markersize=0.5, color='gray')
                ln_med, = self.axes[i].plot(self.xdata_med[i], self.ydata_med[i], 'ro', markersize=1)
                self.ln[i] = ln
                self.ln_med[i] = ln_med
            else:
                # 8 lines per graph: 4 for raw data, 4 for medians
                colors=['red', 'green', 'cyan', 'blue']
                self.ln[i] = []
                self.ln_med[i] = []
                self.ydata[i] = [[] for _ in range(4)]
                self.ydata_med[i] = [[] for _ in range(4)]
                for j in range(4):
                    ln, = self.axes[i].plot(self.xdata[i], self.ydata[i][j], 'o', markersize=0.5, color='gray')
                    ln_med, = self.axes[i].plot(self.xdata_med[i], self.ydata_med[i][j], 'o', markersize=1, color=colors[j])
                    self.ln[i].append(ln)
                    self.ln_med[i].append(ln_med)
        if self.include_radial:
            self.axes[2].set_ylim(-180, 180)
            self.axes[2].set_yticks([-180, -90, 0, 90, 180])
            self.axes[2].set_xlim(0, 60)

        self.axes[0].set_xlim(0, 60)
        self.axes[0].set_ylim(-1, 3)
        self.axes[1].set_xlim(0, 60)
        self.axes[1].set_ylim(15, 25)

    def animate(self):
        return animation.FuncAnimation(
            fig=self.fig, func=self.up, interval=10)

    def update_median(self, median_window_size):
        for graph in range(self.num_graphs):
            self.xdata_med[graph], self.ydata_med[graph] = self.anem_processor_owner.update_medians(median_window_size,
                                                                                                    "strip", graph)

    def up(self, f):
        for g in range(self.num_graphs):
            # Take in input from buffer
            x = 0
            buf_len = len(self.anem_processor_owner.strip_graph_buffer[g])
            med_buf_len = len(self.anem_processor_owner.strip_graph_buffer_med[g])
            for i in range(buf_len):
                (x, y) = self.anem_processor_owner.strip_graph_buffer[g][i]
                self.xdata[g].append(x)
                if self.is_duct:
                    for j in range(4):
                        self.ydata[g][j].append(y[j])
                else:
                    self.ydata[g].append(y)
            for i in range(med_buf_len):
                (x_med, y_med) = self.anem_processor_owner.strip_graph_buffer_med[g][i]
                self.xdata_med[g].append(x_med)
                if self.is_duct:
                    for j in range(4):
                        self.ydata_med[g][j].append(y_med[j])
                else:
                    self.ydata_med[g].append(y_med)

                if np.max(y_med) > self.ymax[g]:
                    self.ymax[g] = np.max(y_med)
                    self.ymax_index[g] = (x_med, len(self.xdata[g]))
                if np.min(y_med) < self.ymin[g]:
                    self.ymin[g] = np.min(y_med)
                    self.ymin_index[g] = (x_med, len(self.xdata[g]))

            # Clear processed elements from buffer
            self.anem_processor_owner.strip_graph_buffer[g] = self.anem_processor_owner.strip_graph_buffer[g][buf_len:]
            self.anem_processor_owner.strip_graph_buffer_med[g] = self.anem_processor_owner.strip_graph_buffer_med[g][med_buf_len:]

            # Scale axes
            try:
                # Rescale y axis if min or max y value has gone out of frame. again, _index holds tuples (time, index)
                if x - GRAPH_MAX_X_WINDOW > self.ymin_index[g][0]:
                    if self.is_duct:
                        val = self.ymax[g]
                        index = 0
                        for i in range(4):
                            v, ind = find_min_max_index(self.ydata_med[g][i], self.ymin_index[g][1] + 1, True)
                            if v < val:
                                val = v
                                index = ind
                    else:
                        val, index = find_min_max_index(self.ydata_med[g], self.ymin_index[g][1] + 1, True)
                    self.ymin[g] = val
                    self.ymin_index[g] = (self.xdata[g][index], index)
                if x - GRAPH_MAX_X_WINDOW > self.ymax_index[g][0]:
                    if self.is_duct:
                        val = self.ymin[g]
                        index = 0
                        for i in range(4):
                            v, ind = find_min_max_index(self.ydata_med[g][i], self.ymax_index[g][1] + 1, True)
                            if v > val:
                                val = v
                                index = ind
                    else:
                        val, index = find_min_max_index(self.ydata_med[g], self.ymax_index[g][1] + 1, False)
                    self.ymax[g] = val
                    self.ymax_index[g] = (self.xdata[g][index], index)
            except IndexError:
                pass

            xminview, xmaxview = self.axes[g].get_xlim()
            if x > xmaxview:
                self.axes[g].set_xlim(max(x - GRAPH_MAX_X_WINDOW, 0), x)
                self.draw()
            if self.scale_y_axes[g]:
                ylim_min = 1.5 * self.ymin[g] if self.ymin[g] < 0 else 0
                self.axes[g].set_ylim(ylim_min, 1.5 * self.ymax[g])
                self.draw()

            # Set data for lines
            if self.is_duct:
                for i in range(4):
                    self.ln[g][i].set_data(self.xdata[g], self.ydata[g][i])
                    self.ln_med[g][i].set_data(self.xdata_med[g], self.ydata_med[g][i])
            else:
                self.ln[g].set_data(self.xdata[g], self.ydata[g])
                self.ln_med[g].set_data(self.xdata_med[g], self.ydata_med[g])

        return self.ln


class ToggleableGraph(FigureCanvas):

    # Initialize a toggleable, real-time graph that pulls from toggle_graph_buffer_y at the given inbuf_index.
    # Initially, this shows velocity (and median velocity) on a given path. When toggled, it shows the relative phases
    # on both directions of the path as well as the absolute phase of one direction.
    def __init__(self, app_window, anem_processor_owner, paths, inbuf_index, parent=None, width=5, height=4, dpi=100):
        self.aw = app_window
        self.anem_processor_owner = anem_processor_owner
        self.paths = paths
        self.inbuf_index = inbuf_index
        self.start_time = time.time()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.suptitle("Velocity on path " + self.data_label())
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("time since start (s)")
        self.axes.set_ylabel("velocity (m/s)")
        self.is_debug = False

        self.xdata = []
        self.ydata_vel = []
        self.xdata_vel_median = []
        self.ydata_vel_median = []
        self.ydata_phase_ab = []
        self.ydata_phase_ba = []
        self.ydata_abs_phase_ab = []
        self.ymin_vel = 0
        self.ymax_vel = 0
        self.ymin_phase = 0
        self.ymax_phase = 0
        self.ymin_vel_index = (0, 0)   # index, x-value
        self.ymax_vel_index = (0, 0)   # index, x-value
        self.ymin_phase_index = (0, 0) # index, x-value
        self.ymax_phase_index = (0, 0) # index, x-value
        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.animation = self.animate()

    def data_label(self):
        return str(self.paths[self.inbuf_index][0]) + " to " + str(self.paths[self.inbuf_index][1])

    def compute_initial_figure(self):
        self.ln, = self.axes.plot(self.xdata, self.ydata_vel, 'o', markersize=1)
        self.ln_median, = self.axes.plot(self.xdata_vel_median, self.ydata_vel_median, 'ro', markersize=1)
        self.ln2, = self.axes.plot(self.xdata, self.ydata_phase_ba, 'ro',
                                   markersize=1)  # line to be used for relative phase when toggled
        self.ln3, = self.axes.plot(self.xdata, self.ydata_abs_phase_ab, 'yo',
                                   markersize=1)  # line to be used for absolute phase when toggled

    def animate(self):
        return animation.FuncAnimation(
            fig=self.fig, func=self.up, interval=10)

    def toggle(self):
        self.is_debug = not self.is_debug
        if self.is_debug:
            self.fig.suptitle(
                "Relative phase on path " + self.data_label())
            self.axes.set_ylabel("Phase (degrees)")
        else:
            self.fig.suptitle(
                "Velocity on path " + self.data_label())
            self.axes.set_ylabel("velocity (m/s)")

    def update_median(self, median_window_size):
        self.xdata_vel_median, self.ydata_vel_median = self.anem_processor_owner.update_medians(median_window_size, "toggle", self.inbuf_index)

    def up(self, f):
        if len(self.anem_processor_owner.toggle_graph_buffer[self.inbuf_index]) == 0:
            return self.ln
        # if not self.anem_processor_owner.is_calibrating:
        #     self.aw.statusBar().clearMessage()

        x = 0
        val_buffer_len = len(self.anem_processor_owner.toggle_graph_buffer[self.inbuf_index])
        median_buffer_len = len(self.anem_processor_owner.toggle_graph_buffer_med[self.inbuf_index])

        # print(self.anem_processor_owner.toggle_graph_buffer)

        for i in range(val_buffer_len):
            (x, (y_phase_ab, y_phase_ba, y_abs_phase_ab, y_vel)) = self.anem_processor_owner.toggle_graph_buffer[self.inbuf_index][i]
            self.xdata.append(x)
            self.ydata_phase_ab.append(y_phase_ab)
            self.ydata_phase_ba.append(y_phase_ba)
            self.ydata_abs_phase_ab.append(y_abs_phase_ab)
            self.ydata_vel.append(y_vel)
            if y_phase_ab > self.ymax_phase:
                self.ymax_phase = y_phase_ab
                self.ymax_phase_index = (x, len(self.xdata))
            if y_phase_ab < self.ymin_phase:
                self.ymin_phase = y_phase_ab
                self.ymin_phase_index = (x, len(self.xdata))
            if y_phase_ba > self.ymax_phase:
                self.ymax_phase = y_phase_ba
                self.ymax_phase_index = (x, len(self.xdata))
            if y_phase_ba < self.ymin_phase:
                self.ymin_phase = y_phase_ba
                self.ymin_phase_index = (x, len(self.xdata))
            if y_vel > self.ymax_vel:
                self.ymax_vel = y_vel
                self.ymax_vel_index = (x, len(self.xdata))
            if y_vel < self.ymin_vel:
                self.ymin_vel = y_vel
                self.ymin_vel_index = (x, len(self.xdata))
        
        for i in range(median_buffer_len): 
            (x_med, y_med) = self.anem_processor_owner.toggle_graph_buffer_med[self.inbuf_index][i]
            self.xdata_vel_median.append(x_med)
            self.ydata_vel_median.append(y_med)

        self.anem_processor_owner.toggle_graph_buffer[self.inbuf_index] = self.anem_processor_owner.toggle_graph_buffer[self.inbuf_index][val_buffer_len:]
        self.anem_processor_owner.toggle_graph_buffer_med[self.inbuf_index] = self.anem_processor_owner.toggle_graph_buffer_med[self.inbuf_index][median_buffer_len:]

        # set view frame so you can see the entire line
        # TODO: This should be synchronized across all similar graphs.

        if self.is_debug:
            # Adjust ymin, ymax to be the largest and smallest y values in the current viewing window
            try:
                if x - GRAPH_MAX_X_WINDOW > self.ymin_phase_index[0]:
                    search_start = self.ymin_phase_index[1] + 1
                    self.ymin_phase = self.y_phase_ab[search_start]
                    self.ymin_phase_index = (self.xdata[search_start], search_start)
                    for i in range(search_start, len(self.y_phase_ab)):
                        if self.y_phase_ab[i] < self.ymin_phase:
                            self.ymin_phase = self.y_phase_ab[i]
                            self.ymin_phase_index = (self.xdata[i], i)
                        if self.y_phase_ba[i] < self.ymin_phase:
                            self.ymin_phase = self.y_phase_ba[i]
                            self.ymin_phase_index = (self.xdata[i], i)
                if x - GRAPH_MAX_X_WINDOW > self.ymax_phase_index[0]:
                    self.ymax_phase = self.y_phase_ab[search_start]
                    self.ymax_phase_index = (self.xdata[search_start], search_start)
                    for i in range(search_start, len(self.y_phase_ab)):
                        if self.y_phase_ab[i] > self.ymax_phase:
                            self.ymax_phase = self.y_phase_ab[i]
                            self.ymax_phase_index = (self.xdata[i], i)
                        if self.y_phase_ba[i] > self.ymax_phase:
                            self.ymax_phase = self.y_phase_ba[i]
                            self.ymax_phase_index = (self.xdata[i], i)
            except IndexError:
                # edge case. Probably don't worry about this.
                pass

            ylim_min = 1.5 * self.ymin_phase if self.ymin_phase < 0 else 0
            self.axes.set_ylim(ylim_min, 1.5 * self.ymax_phase)
            self.draw()

            self.ln.set_data(self.xdata, self.ydata_phase_ab)
            self.ln_median.set_data([], [])
            self.ln2.set_data(self.xdata, self.ydata_phase_ba)
            self.ln3.set_data(self.xdata, self.ydata_abs_phase_ab)
            # todo fix
        else:
            # Adjust ymin, ymax to be the largest and smallest y values in the current viewing window
            try:
                if x - GRAPH_MAX_X_WINDOW > self.ymin_vel_index[0]:
                    search_start = self.ymin_vel_index[1] + 1
                    self.ymin_vel = self.ydata_vel[search_start]
                    self.ymin_vel_index = (self.xdata[search_start], search_start)
                    for i in range(search_start, len(self.ydata_vel)):
                        if self.ydata_vel[i] < self.ymin_vel:
                            self.ymin_vel = self.ydata_vel[i]
                            self.ymin_vel_index = (self.xdata[i], i)
                if x - GRAPH_MAX_X_WINDOW > self.ymax_vel_index[0]:
                    search_start = self.ymax_vel_index[1] + 1
                    self.ymax_vel = self.ydata_vel[search_start]
                    self.ymax_vel_index = (self.xdata[search_start], search_start)
                    for i in range(search_start, len(self.ydata_vel)):
                        if self.ydata_vel[i] > self.ymax_vel:
                            self.ymax_vel = self.ydata_vel[i]
                            self.ymax_vel_index = (self.xdata[i], i)
            except IndexError:
                pass

            ylim_min = 1.5 * self.ymin_vel if self.ymin_vel < 0 else 0
            self.axes.set_ylim(ylim_min, 1.5 * self.ymax_vel)
            self.draw()

            self.ln.set_data(self.xdata, self.ydata_vel)
            self.ln_median.set_data(self.xdata_vel_median, self.ydata_vel_median)
            self.ln2.set_data([], [])
            self.ln3.set_data([], [])

        xminview, xmaxview = self.axes.get_xlim()
        if x > xmaxview:
            self.axes.set_xlim(max(x - GRAPH_MAX_X_WINDOW, 0), x)
            self.draw()
        self.draw()

        # return [self.ln, self.ln2, self.ln3]
        return self.ln


class GeneralGraph(FigureCanvas):

    # Initialize a non-toggle, real-time graph that pulls from general_graph_buffer_y at the given inbuf_index.
    # Title, ylabel, yrange, and yticks can be customized.
    def __init__(self, app_window, anem_processor_owner, inbuf_index, parent=None,
                 width=5, height=4, dpi=100, title=None, ylabel=None, yrange=None, yticks=None):
        self.aw = app_window
        self.anem_processor_owner = anem_processor_owner
        self.inbuf_index = inbuf_index
        self.start_time = time.time()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if title is None:
            char = 'x'
            if inbuf_index == 1:
                char = 'y'
            elif inbuf_index == 2:
                char = 'z'
            title = "Velocity on " + char + " axis"
        if ylabel is None:
            ylabel = "velocity (m/s)"
        self.title = title
        self.fig.suptitle(title)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("time since start (s)")
        self.axes.set_ylabel(ylabel)

        self.xdata = []
        self.ydata = []
        self.xdata_med = []
        self.ydata_med = []
        self.ymin = 0
        self.ymax = 0
        self.ymin_index = (0, 0)
        self.ymax_index = (0, 0)
        self.update_ylim = (yrange is None)  # If a custom y range is set, don't update the y range on new data
        self.yrange = yrange
        self.compute_initial_figure(yrange, yticks)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.animation = self.animate()

    def data_label(self):
        return self.title

    def compute_initial_figure(self, yrange, yticks):
        self.ln, = self.axes.plot(self.xdata, self.ydata, 'o', markersize=1)
        self.ln_med, = self.axes.plot(self.xdata_med, self.ydata_med, 'ro', markersize=1)
        if yrange is not None:
            self.axes.set_ylim(yrange[0], yrange[1])
            # self.axes.autoscale(False)
        if yticks is not None:
            self.axes.yaxis.set_ticks(yticks)

    def animate(self):
        return animation.FuncAnimation(
            fig=self.fig, func=self.up, interval=10)

    def update_median(self, median_window_size):
        self.xdata_med, self.ydata_med = self.anem_processor_owner.update_medians(median_window_size, "general", self.inbuf_index)

    def up(self, f):
        # if not self.anem_processor_owner.is_calibrating:
        #     self.aw.statusBar().clearMessage()
        x = 0
        val_buffer_len = len(self.anem_processor_owner.general_graph_buffer[self.inbuf_index])
        median_buffer_len = len(self.anem_processor_owner.general_graph_buffer_med[self.inbuf_index])
        for i in range(val_buffer_len):
            (x, y) = self.anem_processor_owner.general_graph_buffer[self.inbuf_index][i]
            self.xdata.append(x)
            self.ydata.append(y)
            if y > self.ymax:
                self.ymax = y
                self.ymax_index = (x, len(self.xdata))
            if y < self.ymin:
                self.ymin = y
                self.ymin_index = (x, len(self.xdata))

        for i in range(median_buffer_len):
            (x_med, y_med) = self.anem_processor_owner.general_graph_buffer_med[self.inbuf_index][i]
            self.xdata_med.append(x_med)
            self.ydata_med.append(y_med)

        # Clear from the buffer the elements just processed
        self.anem_processor_owner.general_graph_buffer[self.inbuf_index] = self.anem_processor_owner.general_graph_buffer[self.inbuf_index][val_buffer_len:]
        self.anem_processor_owner.general_graph_buffer_med[self.inbuf_index] = self.anem_processor_owner.general_graph_buffer_med[self.inbuf_index][median_buffer_len:]

        # set view frame so you can see the entire line
        # TODO: This should be synchronized across all similar graphs.
        try: 
            if x - GRAPH_MAX_X_WINDOW > self.ymin_index[0]:
                search_start = self.ymin_index[1] + 1
                self.ymin = self.ydata[search_start]
                self.ymin_index = (self.xdata[search_start], search_start)
                for i in range(search_start, len(self.ydata)):
                    if self.ydata[i] < self.ymin:
                        self.ymin = self.ydata[i]
                        self.ymin_index = (self.xdata[i], i)
            if x - GRAPH_MAX_X_WINDOW > self.ymax_index[0]:
                search_start = self.ymax_index[1] + 1
                self.ymax = self.ydata[search_start]
                self.ymax_index = (self.xdata[search_start], search_start)
                for i in range(search_start, len(self.ydata)):
                    if self.ydata[i] > self.ymax:
                        self.ymax = self.ydata[i]
                        self.ymax_index = (self.xdata[i], i)
        except IndexError:
            pass

        ylim_min = 1.5 * self.ymin if self.ymin < 0 else 0
        self.axes.set_ylim(ylim_min, 1.5 * self.ymax)
        self.draw()

        xminview, xmaxview = self.axes.get_xlim()
        if x > xmaxview:
            self.axes.set_xlim(max(x - GRAPH_MAX_X_WINDOW, 0), x)
            self.draw()
        self.ln.set_data(self.xdata, self.ydata)
        self.ln_med.set_data(self.xdata_med, self.ydata_med)
        # return self.ln,
        return self.ln


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def strformat_double(num):
    if num >= 10:
        return "{:.1f}".format(num)
    else:
        return "{:.2f}".format(num)


def find_min_max_index(data, start_index, find_min):
    search_data = data[start_index:]
    if find_min:
        val = min(search_data)
    else:
        val = max(search_data)
    index = search_data.index(val) + start_index
    return val, index
