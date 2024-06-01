import cv2
import numpy as np
from ultralytics import YOLO
from object_detection import colorChange,size_changer

# 비디오 캡처
if __name__ == '__main__':
    cap = cv2.VideoCapture('data/testvid2.mp4')
    # musicpath='music\Diviners feat. Contacreast - Tropic Love [NCS Release].mp3'
    # tempo, beat_times = extract_beat_timing(musicpath)
    mode=0
    # print(tempo, beat_times)

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
        results = model.predict(img)
        img= size_changer(img, results, 0.5)  # Change the size of the object in the image
        cv2.waitKey(1)
        
        # img = filter_image_with_lut(img)  # Apply LUT filter to the image
        cv2.imshow('veatbision', img)
    cap.release()
    cv2.destroyAllWindows()