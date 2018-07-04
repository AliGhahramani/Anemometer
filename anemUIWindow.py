# Holds classes for starting up a UI window for tracking anemometer readings

from __future__ import unicode_literals
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
import numpy as np
import time
import os

GRAPH_MAX_X_WINDOW = 200
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

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

        # Create layout
        # l = QtWidgets.QVBoxLayout(self.main_widget)
        l = QtWidgets.QGridLayout(self)
        
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
                                          str(anemometer_id) + '.tsv')
        self.dump_graph_button.clicked.connect(self.on_dump_graph_data_click)

        self.median_window_textbox = QtWidgets.QLineEdit(self)
        self.median_window_textbox.setText('5') # default median window size is 5
        self.median_window_textbox.setMaxLength(4)
        self.median_window_label = QtWidgets.QLabel('Median window size', self)
        self.median_window_button = QtWidgets.QPushButton('SET', self)
        self.median_window_button.clicked.connect(self.on_set_median_window_click)

        l.addWidget(self.debug_toggle_button, 0, 0)
        l.addWidget(self.dump_data_button, 1, 0)
        l.addWidget(self.dump_graph_button, 2, 0)
        l.addWidget(self.median_window_label, 4, 0)
        l.addWidget(self.median_window_textbox, 5, 0)
        l.addWidget(self.median_window_button, 6, 0)

        # Add graphs
        # TODO: Probably do this with subplots instead of multiple plots..
        self.toggle_graphs = []
        self.general_graphs = []
        for i in range(len(self.paths)):
            g = ToggleableGraph(self, anem_processor_owner, self.paths, i, self.main_widget, width=5, height=4, dpi=100)
            self.general_graphs.append(g)
            # l.addWidget(g)
            l.addWidget(g, i % 2, i//2 + 1)

        if is_duct:
            # Graphs for temperature
            for i in range(len(self.paths)):
                path_str = str(self.paths[i][0]) + " to " + str(self.paths[i][1])
                g = GeneralGraph(self, anem_processor_owner, i, self.main_widget, width=5, height=4, dpi=100,
                                     title="Temp on path " + path_str, ylabel="temperature (C)", yrange=(21, 25))
                self.general_graphs.append(g)
                l.addWidget(g, 2 + i % 2, i//2 + 1)
        else:
            # Graphs for vx, vy, vz, m, theta, phi
            vx = GeneralGraph(self, anem_processor_owner, 0, self.main_widget, width=5, height=4, dpi=100)
            vy = GeneralGraph(self, anem_processor_owner, 1, self.main_widget, width=5, height=4, dpi=100)
            vz = GeneralGraph(self, anem_processor_owner, 2, self.main_widget, width=5, height=4, dpi=100)
            m = GeneralGraph(self, anem_processor_owner, 3, self.main_widget, width=5, height=4, dpi=100,
                                 title="Wind speed", ylabel="speed (m/s)")
            theta = GeneralGraph(self, anem_processor_owner, 4, self.main_widget, width=5, height=4, dpi=100,
                                     title="Radial angle", ylabel="radial angle (degrees)", yrange=(-220, 220),
                                     yticks=[-180, -90, 0, 90, 180])
            phi = GeneralGraph(self, anem_processor_owner, 5, self.main_widget, width=5, height=4, dpi=100,
                                   title="Vertical angle", ylabel="vertical angle (degrees)", yrange=(-100, 100),
                                   yticks=[-90, -60, -30, 0, 30, 60, 90])
            self.general_graphs.extend([vx, vy, vz, m, theta, phi])
            l.addWidget(vx, 2, 1)
            l.addWidget(vy, 2, 2)
            l.addWidget(vz, 2, 3)
            l.addWidget(m, 3, 1)
            l.addWidget(theta, 3, 2)
            l.addWidget(phi, 3, 3)

        self.main_widget.setFocus()
        # self.setCentralWidget(self.main_widget)
        # if self.anem_processor_owner.is_calibrating:
        #     self.statusBar().showMessage("CALIBRATING. Please wait")

        print("CREATED WINDOW " + self.anemometer_id)

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
            return;
        if self.anem_processor_owner.median_window_size != new_median_window_size:
            self.anem_processor_owner.median_window_size = new_median_window_size
            for graph in self.toggle_graphs:
                graph.update_median(new_median_window_size)
            for graph in self.general_graphs:
                graph.update_median(new_median_window_size)

    def on_dump_graph_data_click(self):
        f = open(resource_path('anemometer_graph_data_' + str(self.anemometer_id) + '.tsv'), 'w')

        # write column labels
        f.write("time since start")
        for graph in self.toggle_graphs:
            f.write("\t" + str(graph.data_label()))
        for graph in self.general_graphs:
            f.write("\t" + str(graph.data_label()))
        f.write("\n")

        # write all data lines
 #       for i in range(len(self.toggle_graphs[0].xdata)):
 #           f.write(str(self.toggle_graphs[0].xdata[i]))  # time since start
 #           for graph in self.toggle_graphs:
 #               if i < len(graph.ydata_vel):
 #                   f.write("\t" + str(graph.ydata_vel[i]))
 #           for graph in self.general_graphs:
 #               if i < len(graph.ydata):
 #                   f.write("\t" + str(graph.ydata[i]))
 #           f.write("\n")

        f.close()


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
        self.xdata_vel_median, self.ydata_vel_median = self.anem_processor_owner.update_medians(median_window_size, True, self.inbuf_index)

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
                if (x - GRAPH_MAX_X_WINDOW > self.ymin_phase_index[0]):
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
                if (x - GRAPH_MAX_X_WINDOW > self.ymax_phase_index[0]):
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
                if (x - GRAPH_MAX_X_WINDOW > self.ymin_vel_index[0]):
                    search_start = self.ymin_vel_index[1] + 1
                    self.ymin_vel = self.ydata_vel[search_start]
                    self.ymin_vel_index = (self.xdata[search_start], search_start)
                    for i in range(search_start, len(self.ydata_vel)):
                        if self.ydata_vel[i] < self.ymin_vel:
                            self.ymin_vel = self.ydata_vel[i]
                            self.ymin_vel_index = (self.xdata[i], i)
                if (x - GRAPH_MAX_X_WINDOW > self.ymax_vel_index[0]):
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
        self.xdata_med, self.ydata_med = self.anem_processor_owner.update_medians(median_window_size, False, self.inbuf_index)

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
            if (x - GRAPH_MAX_X_WINDOW > self.ymin_index[0]):
                search_start = self.ymin_index[1] + 1
                self.ymin = self.ydata[search_start]
                self.ymin_index = (self.xdata[search_start], search_start)
                for i in range(search_start, len(self.ydata)):
                    if self.ydata[i] < self.ymin:
                        self.ymin = self.ydata[i]
                        self.ymin_index = (self.xdata[i], i)
            if (x - GRAPH_MAX_X_WINDOW > self.ymax_index[0]):
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
