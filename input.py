
import sys, time, os

# CAREFUL! Using threading (and thus, timers) here messes up piping into input...
with open("newdata2.txt") as f:
    for line in f: 
        print line,
        sys.stdout.flush()
        i = 0
        for j in range(6000000):
            i+=1
