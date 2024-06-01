import cv2 as cv
import numpy as np

def get_beat_effect_coefficient(framenum,frame,beat_times,tempo):
    
    beat_pulse_list = [i for i in np.linspace(0, 1, 50)] +[i for i in np.linspace(1, 0, 50)]
    beat_amount_index=fit_beat_frame_time(framenum,frame,beat_times,tempo) 
    print(beat_amount_index,beat_pulse_list[beat_amount_index])
    return beat_pulse_list[beat_amount_index]

def fit_beat_frame_time(framenum,frame,beat_times,tempo):
    
    frameduration=1/frame
    beattime=find_nearest_beat_time(framenum,beat_times,frameduration)
    deviation = framenum*frameduration-beattime
    effect_duration=60/tempo
    print(deviation)
    
    scaled_deviation = ((effect_duration/2 + deviation)/effect_duration)*100
    scaled_deviation = int(scaled_deviation)
    print(scaled_deviation)
    beat_amount_index = max(0, min(99, scaled_deviation))
    return beat_amount_index
    
def find_nearest_beat_time(framenum, beat_times, frameduration):
    target_time = frameduration * framenum
    nearest_time = min(beat_times, key=lambda x: abs(x - target_time))
    return nearest_time



if __name__ == '__main__':
    print()
    

       

