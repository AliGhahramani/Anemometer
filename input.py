
import os,sys, time
# CAREFUL! Using threading (and thus, timers) here messes up piping into input...
read_frequency = 10 # 2 hz or 6 hz
time_interval = 1/read_frequency

def main():
        with open("anemometer_raw_data_room.txt") as f:
            for line in f:  
                    print (line),
                    try:
                        sys.stdout.flush()        
                        time.sleep(time_interval)
                    except Exception:
                       
                        os._exit(1)
        print("Testing data End.................")
        os._exit(1)
if __name__ == "__main__":
    main()
