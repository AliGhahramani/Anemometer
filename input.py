
import sys, time, os

# CAREFUL! Using threading (and thus, timers) here messes up piping into input...
with open("temp_jump.txt") as f:
    for line in f: 
        print line,
        sys.stdout.flush()
        i = 0
        for j in range(600000):
            i+=1
