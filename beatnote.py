import cv2 as cv
import numpy as np

def get_beat_effect_coefficient(framenum,beat_times,tempo):
    beat_pulse_list = [0] + [i/99 for i in range(1, 99)] + [0]
    beat_amount_index=fit_beat_frame_time(framenum,beat_times,tempo)
    return beat_pulse_list[beat_amount_index]

def fit_beat_frame_time(framenum,beat_times,tempo):
    frame=30
    frameduration=1/frame
    beattime=find_nearest_beat_time(framenum,beat_times,frameduration)
    deviation = framenum*(1/frameduration)-beattime
    effect_duration=60/tempo
    
    scaled_deviation = ((effect_duration/2 + deviation)/effect_duration)*100
    scaled_deviation = int(scaled_deviation)
    beat_amount_index = max(0, min(100, scaled_deviation))
    return beat_amount_index
    
def find_nearest_beat_time(framenum, beat_times, frameduration):
    target_time = frameduration * framenum
    nearest_time = min(beat_times, key=lambda x: abs(x - target_time))
    return nearest_time


