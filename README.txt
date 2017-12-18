AnemUI Readme
=========================================================
The anemometer UI is built in pyqt5, with matplotlib embedded in widgets. It processes the stream of data from the executable, buffering until each reading is complete and processed, then displays the reading in real time in matplotlib graphs.

Note that with the addition of 3 point noise discrimination, the actively buffered or currently being processed reading is named the 'next' reading (perhaps confusingly), and only after this reading is processed can the 'current' reading be validated and added to the UI.


Setup and usage
=========================================================
You will need to install python3, pyqt5, and matplotlib. 
You will need anemUI.py, reading.py, anemomteer-28589, anemomteer-master, and optionally input.py and some input file all in the same directory. To run, set is_duct in anemUI.py to True or False depending on if you are processing data from the duct anemometer or the room anemometer, and save. Then in command line, navigate to the directory and run
	python3 anemUI.py

To use an input file, in anemUI.py, replace
 	p = Popen([executable_path], stdout=PIPE)
with
    p = Popen(["python", "input.py"], stdout=PIPE) 
and in input.py, replace the file name like so:
	with open("YOUR_FILE.txt") as f:

If, when using an input file, the data is processed too fast and appears clumped, you can artificially slow the input of data by uncommenting the for loops in input.py.

When running the program, you can hit the toggle button to display phase diagrams to debug, or you can hit the save data button to dump out all data read to anemometer_data.txt in the directory. This will overwrite any previous saves if the name was not changed.


Algorithm overview
=========================================================
With 4 nodes, there are a total of 16 paths (including self loops). Every half second, for each path, we get 16 magnitude (M), imaginary (Q), and real readings (I), and this whole collection is considered one data point and stored in a Reading (defined in readings.py). Our goal is to extract a velocity for each path, and then consolidate these path velocities into a meaningful form; in the case of the duct anemometer, we combine opposite path velocities to conclude the true velocity along that path, and for the room anemometer, we further extrapolate velocities along the x, y, and z axes, as well as the overall velocity magnitude, and direction transormed into spherical coordinates. 

To transform the I, Q, M values for a Reading to velocities, there are a few steps. As of right now, we are using a phase-only calculation. Below, outlined for the i-th Reading:
	For each path:
		abs_phase(i) = absolute phase of i-th Reading = quadrant arctangent(I, Q), of the I and Q values at the index 2 before the maximum magnitude (found during calibration)
		delta = abs_phase(i) - abs_phase(i-1)
		wrapped_delta = delta if abs(delta) < 180, else, -1 * sign(delta) * (360 - abs(delta))
		
		If noise, drop this point:
			rel_phase(i) = rel_phase(i-1)
			abs_phase(i) = abs_phase(i-1)
		else:
			rel_phase(i) = rel_phase(i-1) + wrapped_delta

As of right now, a point is considered noise if it is within the outlier threshhold, or if it fails the 3 point noise discrimination criteria, explained below.
We can very easily transform this relative phase to velocity on each path like so:
	d = length of path
	time of flight a to b = relative phase a to b / (360 * 180000)
	time of flight b to a = relative phase b to a / (360 * 180000)
	velocity from a to b = d / (d / 343 + time of flight a to b)
	velocity from b to a = d / (d / 343 + time of flight b to a)
	relative velocity = (velocity from a to b - velocity from b to a) / 2 / cos(pi/4)

For the room anemometer, we can further transform this velocity per path to velocity per axis by accounting for the tetrahedron geometry of the room anemometer, to get the following weights per path(assuming node 1 is at the bottom):
        
 	Velocity on x-axis (vx):
    0-1= Sin(30)xSin(30)
    0-2= Sin(60)
    0-3= 0
    1-2= Sin(30)
    1-3= -Sin(30)xSin(30)
    2-3= -Sin(60)

    Velocity on y-axis (vy):
    0-1= Sin(60)xSin(30)
    0-2= Sin(30)
    0-3= 1
    1-2= 0
    1-3= Sin(60)xSin(30)
    2-3= Sin(30)

    Velocity on z-axis (vz):
    0-1= -sin(60)
    0-2= 0
    0-3= 0
    1-2= sin(60)
    1-3= sin(60)
    2-3= 0

And of course, we can easily transform this to spherical coordinates where
	magnitude = sqrt(vx^2 + vy^2 + vz^2)
	theta = arctan2(vy, vx)
	phi = arcsin(vz/m)
with the small caveat that we artificially set theta and phi to 0 if the magnitude is small for graph readability, as the directions are very noisy when there is little to no air flow.


Outlier threshhold
=========================================================
We drop any point within X degrees of a 180 degree shift, to filter out dangerously (in the context of derailing our algorithm) large noise.
If we only attempt to find the closest rotation of an absolute phase to determine its relative phase, we may get undesired jumps in phase. For example, say we have a stream of absolute phases -20, 10, 0, 20, -170, -20, a closest rotation interpretation would produce the relative phases -20, 10 (+30), 0 (-10), 20 (+20), 190 (+170), 340 (+150). We can interpret this as not ruling out a dangerous noisy point, where the noise is near the 180 degree decision threshhold, so instead, we reject all points within a certain degree of 180. If for example, we set the threshhold at 50, then we allow a maximum of 130 degree phase shift. So, the -170 point will be discarded as noise, and the following point -20 will be compared to the previous point 20, and thus be unaffected.


3 point noise discrimination
=========================================================
We drop any point where the difference between the change between the next point and this point, and this point and the previous point, is greater than X. Essentially, the previous and next point are close together, but this point is far off.
Suppose our outlier threshhold is 50 degrees. Let a collection of consecutive relative phases in our stream be 20, 140, -20, -40, etc. Here our algorithm should deduce that 140 is noise, and the correct stream is 20, -20, -40. Unfortunately, if we use only an outlier threshhold to remove large noise, then we will erroneously conclude that 140 is valid and -30 is noise and continue to assume the rest of the stream is noise until another point comes within (180-50) of 140. This is especially problematic if the trend is in the other direction, as the algorithm will assume that the phase wrapped and jumped up, rather than gradually decreasing. To combat this, we must also look at the point after 140 to determine 140's validity. Specifically, if the distance between a point and its predecessor and its successor are large, and the predecessor and successor are close, then the current point is likely noise and can be ignored.


Code overview
=========================================================
readings.py holds the data structure for a single Reading, which is a single datapoint (a datapoint is transmitted every half second). 

input.py is an optional, artificial way to pipe in input from a text file as if it's coming in realtime from the executable.

anemUI.py does all the data parsing, signal processing, and displays the UI
Readings are continuously read in and parsed in read_input(). When a Reading is complete, it is processed in process_reading() which calculates the path velocities and adds them to an in buffer (inbuf, and inbuf_consolidated if room anemometer) to be added to the graphs.
The UI (ApplicationWindow) is also set up in another thread, creating each graph (MyToggleCanvas and MyVelocityCanvas) such that each graph knows which in buffer to read from. Each graph then starts an animation, which repeatedly calls their update function (up(self, f)), in which the in buffer is read and added to the graph. In addition, MyToggleCanvas can be toggled between showing the velocity on a path (default) or the relative phases in both directions, and the absolute phase in one direction (toggled) for debugging purposes.



    