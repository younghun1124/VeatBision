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
    video_path = 'data/testvid.mp4'
    cap = cv2.VideoCapture('data/testvid.mp4')
    musicpath='music\될놈\Speo - Make A Stand (feat. Budobo) [NCS Release].mp3'
    tempo, beat_times = extract_beat_timing(musicpath)
    print(beat_times)
    mode=0   
    frame_num=0
    # 모델 로드 합니다
    model = YOLO('yolov8n-seg.pt')
    # 비디오 녹화 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = f'{musicpath.split("/")[-1].split(".")[0]}_{video_path.split("/")[-1].split(".")[0]}.avi'
    target = cv2.VideoWriter()

    # 비디오 녹화 시작
    while True:
        frame_num += 1
        valid, img = cap.read()
        fps = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        waitsec = int(1 / fps * 1000)
        
        if not valid:
            print("영상없음")
            break
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (img.shape[1], img.shape[0]))
        beat_effect_coeff = get_beat_effect_coefficient(frame_num, beat_times, tempo)
        results = model.predict(img)
        if not target.isOpened():
            h,w,*_=img.shape
            is_color = (img.ndim > 2) and (img.shape[2] > 1)
            cv2.destroyAllWindows()
            
            target.open(output_path, fourcc, fps, (w, h), is_color)
            target.write(img)
        
        key = cv2.waitKey(waitsec) & 0xFF
        if key == ord(' '):
            cv2.waitKey()
        elif key == ord('0'):
            mode = 0
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('q'):
            break
        
        if mode == 1:
            img = colorChange(img, results, beat_effect_coeff)
        elif mode == 2:
            img = size_changer(img, results, beat_effect_coeff)
        
        target.write(img)  # 비디오 녹화
        cv2.imshow('veatbision', img)

    cap.release()
    target.release()
    cv2.destroyAllWindows()