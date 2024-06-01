import time

def read_beat_data(file_path):
    with open(file_path, 'r') as file:
        beat_data = file.readlines()
    return beat_data

def print_beat_at_timings(beat_data):    
    for timing in beat_data:
        time.sleep(float(timing))  # Convert milliseconds to seconds
        print("Beat!")

if __name__ == '__main__':
    file_path = "./beats/dd.csv"  # Replace with the actual file path

    beat_data = read_beat_data(file_path)
    print_beat_at_timings(beat_data)