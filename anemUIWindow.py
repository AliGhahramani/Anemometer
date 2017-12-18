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

MEDIAN_WINDOW_SIZE = 5

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
        l.addWidget(self.debug_toggle_button, 0, 0)
        l.addWidget(self.dump_data_button, 1, 0)
        l.addWidget(self.dump_graph_button, 2, 0)

        # Add graphs
        # TODO: Probably do this with subplots instead of multiple plots..
        self.toggle_graphs = []
        self.velocity_graphs = []
        for i in range(len(self.paths)):
            g = MyToggleCanvas(self, anem_processor_owner, self.paths, i, self.main_widget, width=5, height=4, dpi=100)
            self.toggle_graphs.append(g)
            # l.addWidget(g)
            l.addWidget(g, i % 2, i//2 + 1)

        if is_duct:
            # Graphs for temperature
            for i in range(len(self.paths)):
                path_str = str(self.paths[i][0]) + " to " + str(self.paths[i][1])
                g = MyVelocityCanvas(self, anem_processor_owner, i, self.main_widget, width=5, height=4, dpi=100,
                                     title="Temp on path " + path_str, ylabel="temperature (C)", yrange=(21, 25))
                self.velocity_graphs.append(g)
                l.addWidget(g, 2 + i % 2, i//2 + 1)
        else:
            # Graphs for vx, vy, vz, m, theta, phi
            vx = MyVelocityCanvas(self, anem_processor_owner, 0, self.main_widget, width=5, height=4, dpi=100)
            vy = MyVelocityCanvas(self, anem_processor_owner, 1, self.main_widget, width=5, height=4, dpi=100)
            vz = MyVelocityCanvas(self, anem_processor_owner, 2, self.main_widget, width=5, height=4, dpi=100)
            m = MyVelocityCanvas(self, anem_processor_owner, 3, self.main_widget, width=5, height=4, dpi=100,
                                 title="Speed", ylabel="speed (m/s)")
            theta = MyVelocityCanvas(self, anem_processor_owner, 4, self.main_widget, width=5, height=4, dpi=100,
                                     title="Radial angle", ylabel="radial angle (degrees)", yrange=(-220, 220),
                                     yticks=[-180, -90, 0, 90, 180])
            phi = MyVelocityCanvas(self, anem_processor_owner, 5, self.main_widget, width=5, height=4, dpi=100,
                                   title="Vertical angle", ylabel="vertical angle (degrees)", yrange=(-100, 100),
                                   yticks=[-90, -60, -30, 0, 30, 60, 90])
            self.velocity_graphs.extend([vx, vy, vz, m, theta, phi])
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

    def on_dump_graph_data_click(self):
        f = open('anemometer_graph_data_' + str(self.anemometer_id) + '.tsv', 'w')

        # write column labels
        f.write("time since start")
        for graph in self.toggle_graphs:
            f.write("\t" + str(graph.data_label()))
        for graph in self.velocity_graphs:
            f.write("\t" + str(graph.data_label()))
        f.write("\n")

        # write all data lines
        for i in range(len(self.toggle_graphs[0].xdata)):
            f.write(str(self.toggle_graphs[0].xdata[i]))  # time since start
            for graph in self.toggle_graphs:
                f.write("\t" + str(graph.ydata_vel[i]))
            for graph in self.velocity_graphs:
                f.write("\t" + str(graph.ydata[i]))
            f.write("\n")

        f.close()


class MyToggleCanvas(FigureCanvas):

    # Initialize a toggleable, real-time graph that pulls from inbuf_toggle at the given inbuf_index.
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

    def up(self, f):
        if len(self.anem_processor_owner.inbuf_toggle[self.inbuf_index]) == 0:
            return self.ln
        # if not self.anem_processor_owner.is_calibrating:
        #     self.aw.statusBar().clearMessage()

        x = 0
        for i in range(len(self.anem_processor_owner.inbuf_toggle[self.inbuf_index])):
            x = time.time() - self.start_time
            y_phase_ab, y_phase_ba, y_abs_phase_ab, y_vel = self.anem_processor_owner.inbuf_toggle[self.inbuf_index][i]
            self.xdata.append(x)
            self.ydata_phase_ab.append(y_phase_ab)
            self.ydata_phase_ba.append(y_phase_ba)
            self.ydata_abs_phase_ab.append(y_abs_phase_ab)
            self.ydata_vel.append(y_vel)
            if y_phase_ab > self.ymax_phase:
                self.ymax_phase = y_phase_ab
            if y_phase_ab < self.ymin_phase:
                self.ymin_phase = y_phase_ab
            if y_phase_ba > self.ymax_phase:
                self.ymax_phase = y_phase_ba
            if y_phase_ba < self.ymin_phase:
                self.ymin_phase = y_phase_ba
            if y_vel > self.ymax_vel:
                self.ymax_vel = y_vel
            if y_vel < self.ymin_vel:
                self.ymin_vel = y_vel

            # calculate median (todo: this should be in processor)
            if len(self.xdata) >= MEDIAN_WINDOW_SIZE:
                x_med = np.median(self.xdata[-MEDIAN_WINDOW_SIZE:])
                y_med = np.median(self.ydata_vel[-MEDIAN_WINDOW_SIZE:])
                self.xdata_vel_median.append(x_med)
                self.ydata_vel_median.append(y_med)

        self.anem_processor_owner.inbuf_toggle[self.inbuf_index] = []

        # set view frame so you can see the entire line
        # TODO: This should be synchronized across all similar graphs.
        # TODO: y limits are currently set according to global max and min; with the sliding time window, this doesn't correspond to viewed max and min
        if self.is_debug:
            ylim_min = 1.5 * self.ymin_phase if self.ymin_phase < 0 else 0
            self.axes.set_ylim(ylim_min, 1.5 * self.ymax_phase)
            self.draw()

            self.ln.set_data(self.xdata, self.ydata_phase_ab)
            self.ln_median.set_data([], [])
            self.ln2.set_data(self.xdata, self.ydata_phase_ba)
            self.ln3.set_data(self.xdata, self.ydata_abs_phase_ab)
            # todo fix
        else:
            ylim_min = 1.5 * self.ymin_vel if self.ymin_vel < 0 else 0
            self.axes.set_ylim(ylim_min, 1.5 * self.ymax_vel)
            self.draw()

            self.ln.set_data(self.xdata, self.ydata_vel)
            self.ln_median.set_data(self.xdata_vel_median, self.ydata_vel_median)
            self.ln2.set_data([], [])
            self.ln3.set_data([], [])
        xminview, xmaxview = self.axes.get_xlim()
        if x > xmaxview:
            self.axes.set_xlim(max(x - 200, 0), x)
            self.draw()
        self.draw()

        # return [self.ln, self.ln2, self.ln3]
        return self.ln

class MyVelocityCanvas(FigureCanvas):

    # Initialize a non-toggle, real-time graph that pulls from inbuf_other at the given inbuf_index.
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

    def up(self, f):
        # if not self.anem_processor_owner.is_calibrating:
        #     self.aw.statusBar().clearMessage()

        x = 0
        for y in self.anem_processor_owner.inbuf_other[self.inbuf_index]:
            x = time.time() - self.start_time
            self.xdata.append(x)
            self.ydata.append(y)
            if y > self.ymax:
                self.ymax = y
            if y < self.ymin:
                self.ymin = y

            # calculate median (todo: this should be in processor)
            if len(self.xdata) >= MEDIAN_WINDOW_SIZE:
                x_med = np.median(self.xdata[-MEDIAN_WINDOW_SIZE:])
                y_med = np.median(self.ydata[-MEDIAN_WINDOW_SIZE:])
                print("median ", y_med)
                self.xdata_med.append(x_med)
                self.ydata_med.append(y_med)

        self.anem_processor_owner.inbuf_other[self.inbuf_index] = []

        # set view frame so you can see the entire line
        # TODO: This should be synchronized across all similar graphs.
        if self.update_ylim:
            ylim_min = 1.5 * self.ymin if self.ymin < 0 else 0
            self.axes.set_ylim(ylim_min, 1.5 * self.ymax)
            self.draw()

        xminview, xmaxview = self.axes.get_xlim()
        if x > xmaxview:
            self.axes.set_xlim(max(x - 200, 0), x)
            self.draw()
        self.ln.set_data(self.xdata, self.ydata)
        self.ln_med.set_data(self.xdata_med, self.ydata_med)
        # return self.ln,
        return self.ln
