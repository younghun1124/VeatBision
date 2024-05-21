from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np
import math

# 웹캠 시작
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
    masks = scale_image(masks, img.shape)
    masks = np.moveaxis(masks, -1, 0) # 마스크, (N, H, W)

    return boxes, masks, cls, probs

def overlay(image, mask, color, alpha, resize=None):
    """이미지와 세그멘테이션 마스크를 결합하여 하나의 이미지로 만듭니다.
    
    Params:
        image: 훈련 이미지. np.ndarray,
        mask: 세그멘테이션 마스크. np.ndarray,
        color: 세그멘테이션 마스크 렌더링을위한 색상. tuple[int, int, int] = (255, 0, 0)
        alpha: 세그멘테이션 마스크의 투명도. float = 0.5,
        resize: 제공되는 경우 이미지와 마스크 모두 조정되어 함께 블렌딩됩니다.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: 결합된 이미지. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# YOLO v8 모델 로드
model = YOLO('yolov8n-seg.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    # YOLOv8로 예측
    boxes, masks, cls, probs = predict_on_image(model, img, conf=0.2)
    
    # 원본 이미지 위에 마스크 오버레이
    image_with_masks = np.copy(img)
    for mask_i in masks:
        image_with_masks = overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0.3)

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