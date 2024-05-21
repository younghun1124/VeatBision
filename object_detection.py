from ultralytics import YOLO
import cv2
import numpy as np
import math

# 웹캠 시작
cap = cv2.VideoCapture(0)


# 객체 클래스
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def predict_on_image(model, img, conf):
    result = model(img, conf=conf)[0]

    # 검출
    # result.boxes.xyxy   # xyxy 형식의 바운딩 박스, (N, 4)
    cls = result.boxes.cls.cpu().numpy()    # 클래스, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # 신뢰도 점수, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # xyxy 형식의 바운딩 박스, (N, 4)

    # 세그멘테이션
    masks = result.masks.data.cpu().numpy()     # 마스크, (N, H, W)
    masks = np.moveaxis(masks, 0, -1) # 마스크, (H, W, N)
    
    # 원본 이미지 크기로 마스크 크기 조정
    masks = cv2.resize(masks, (img.shape[1], img.shape[0]))
    masks = np.moveaxis(masks, -1, 0) # 마스크, (N, H, W)

    return boxes, masks, cls, probs


# YOLO v8 모델 로드
model = YOLO('yolov8n-seg.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    # YOLOv8로 예측
    boxes, masks, cls, probs = predict_on_image(model, img, conf=0.2)
    
    # 원본 이미지 위에 마스크 오버레이
    image_with_masks = np.copy(img)

    # 좌표
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # 바운딩 박스 위치
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # 정수 값으로 변환
            
            # put box in cam
            cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # 신뢰도
            confidence = math.ceil((box.conf[0]*100))/100
            print("신뢰도 =>", confidence)

            # 클래스 이름
            cls = int(box.cls[0])
            print("클래스 이름 ==>", classNames[cls])

            # 객체 정보
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255) 
            thickness = 2

            cv2.putText(image_with_masks, classNames[cls], org, font, fontScale, color, thickness)
            
    cv2.imshow('웹캠', image_with_masks)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()