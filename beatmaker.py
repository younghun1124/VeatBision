import csv
import time

def record_timings():
    timings = []      

    while True:
        key = input("Press Enter to record timing (or press 'q' to stop): ")
        if key == "q":
            break
        elif key == "":            
            timings.append(float(time.time()))
            
    timing_intervals = [timings[i+1] - timings[i] for i in range(len(timings)-1)]
    filename = input("Enter the filename to save the timings: ")
    with open('./beats/'+filename+'.csv', "w", newline="") as file:
        writer = csv.writer(file)          
        writer.writerows(zip(timing_intervals))

    print("Timings saved successfully.")

record_timings()