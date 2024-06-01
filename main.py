import cv2
import numpy as np
import beatmaker
from ultralytics import YOLO
from music_extract import extract_beat_timing
from LUTfilter import filter_image_with_lut
from beatreader import print_beat_at_timings
from beatnote import get_beat_effect_coefficient
from object_detection import colorChange,size_changer

# 비디오 캡처
if __name__ == '__main__':
    cap = cv2.VideoCapture('data/testvid.mp4')
    musicpath='music\Diviners feat. Contacreast - Tropic Love [NCS Release].mp3'
    tempo, beat_times = extract_beat_timing(musicpath)
    mode=0
    print(tempo, beat_times)

    # 모델 로드 합니다
    model = YOLO('yolov8n-seg.pt')

    # print_beat_at_timings(beat_times_ms)
    while True:
        
        # Read an image from 'video'
        valid, img = cap.read()
        if not valid:
            print("영상없음")
            break    
        frame_count =int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame count
        beat_effect_coeff= get_beat_effect_coefficient(frame_count,beat_times, tempo)
      
        results = model.predict(img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.waitKey()
            
        elif key == ord('0'):
            mode=0
        elif key == ord('1'):
            mode=1
        elif key == ord('2'):
            mode=2
            
        elif key == ord('q'):
            break

        if mode == 1:
            img = colorChange(img,results, beat_effect_coeff)
        elif mode == 2:
            img = size_changer(img,results, beat_effect_coeff)
        
        # img = filter_image_with_lut(img)  # Apply LUT filter to the image
        cv2.imshow('veatbision', img)
    cap.release()
    cv2.destroyAllWindows()