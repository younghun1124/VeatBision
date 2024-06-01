import cv2
import numpy as np
from music_extract import extract_beat_timing
from LUTfilter import filter_image_with_lut
from beatreader import print_beat_at_timings
import beatmaker
from beatnote import get_beat_effect_coefficient

# 비디오 캡처
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    musicpath='music\Diviners feat. Contacreast - Tropic Love [NCS Release].mp3'
    tempo, beat_times = extract_beat_timing(musicpath)    
    print(tempo, beat_times)
    
    # print_beat_at_timings(beat_times_ms)
    while True:
        
        # Read an image from 'video'
        valid, img = cap.read()
        frame_count =int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame count
        beat_effect_coeff= get_beat_effect_coefficient(frame_count,beat_times, tempo)
        print("Current frame count:", frame_count)
        
        img = filter_image_with_lut(img)  # Apply LUT filter to the image
        if not valid:
            break    
        # Show the image
        cv2.imshow('Video Player', img)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()