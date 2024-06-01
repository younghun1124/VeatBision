import cv2
import numpy as np
import beatmaker
import re
from ultralytics import YOLO
from music_extract import extract_beat_timing
from LUTfilter import filter_image_with_lut
from beatreader import print_beat_at_timings
from beatnote import get_beat_effect_coefficient
from object_detection import colorChange,size_changer

# 비디오 캡처
if __name__ == '__main__':
    video_path = 'data/testvid5.mp4'
    cap = cv2.VideoCapture(video_path)

    # 모델 로드
    model = YOLO('yolov8n-seg.pt')
    musicpath='music\될놈\Speo - Make A Stand (feat. Budobo) [NCS Release].mp3'
    tempo, beat_times = extract_beat_timing(musicpath)
    print(beat_times)
    mode=0   
    frame_num=0
    color_index=0
    # 비디오 녹화 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # '\' 또는 '/'로 분할하고 마지막 요소를 가져옴
    music_name = re.split(r'[\\\/]', musicpath)[-1]
    video_name = re.split(r'[\\\/]', video_path)[-1]

    # '.'으로 분할하고 첫 번째 요소를 가져옴.
    music_name = music_name.rsplit('.',1)[0]
    video_name = video_name.rsplit('.', 1)[0]

    # 출력 파일 이름
    output_path = f'{music_name}_{video_name}.avi'
    target = cv2.VideoWriter()
    ref_key = ord('0')
    # 비디오 녹화 시작
    while True:
        frame_num += 1
        valid, img = cap.read()
        frame = cap.get(cv2.CAP_PROP_FPS)
        waitsec = int(1 / frame * 1000)
        waitsec = max(waitsec, 30)  # Set a minimum wait time of 30 milliseconds
        
        if not valid:
            print("영상없음")
            break
        print(f'frame: {frame}, frame_num: {frame_num}')
        beat_effect_coeff = get_beat_effect_coefficient(frame_num, frame, beat_times, tempo)
        results = model.predict(img)
        if not target.isOpened():
            h,w,*_=img.shape
            is_color = (img.ndim > 2) and (img.shape[2] > 1) 
            target.open(output_path, fourcc, frame, (w, h), is_color)
            target.write(img)
        
        # 모드 선택
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
        elif key == ord('[') or key == ord('{') or beat_effect_coeff >=0.97:
            color_index = (color_index - 1) % 100
        elif key == ord(']') or key == ord('}') or beat_effect_coeff <=0.03:
            color_index = (color_index + 1) % 100
    
        
        # 모드에 따라 이미지 처리
        if mode == 1:
            img = colorChange(img, results, beat_effect_coeff, key, color_index)
        elif mode == 2:
            img = size_changer(img, results, beat_effect_coeff)
        
        # 비디오 녹화
        target.write(img)
        cv2.imshow('veatbision', img)

    target.release()
    cap.release()
    cv2.destroyAllWindows()