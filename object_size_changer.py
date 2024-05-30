import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델을 로드합니다.
model = YOLO('yolov8n-seg.pt')

# 비디오 파일을 엽니다.
cap = cv2.VideoCapture("data/testvid.mp4")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(img)

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for i in range(len(masks)):
        mask = masks[i]
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # 객체만을 포함하는 마스크를 uint8 타입으로 변환
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        # 객체를 원본 이미지에서 추출
        obj = cv2.bitwise_and(img, img, mask=mask_uint8)

        # 경계 상자 추출
        x1, y1, x2, y2 = boxes[i].astype(int)

        # 경계 상자를 사용하여 객체와 마스크 추출
        obj_cropped = obj[y1:y2, x1:x2]
        mask_cropped = mask_uint8[y1:y2, x1:x2]

        # 객체 크기를 두 배로 확대
        obj_large = cv2.resize(obj_cropped, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        mask_large = cv2.resize(mask_cropped, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # 확대된 객체의 크기
        h_large, w_large = obj_large.shape[:2]

        # 원본 이미지에서 확대된 객체의 중심을 탐지된 객체의 중심으로 맞춤
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        start_x = max(0, center_x - w_large // 2)
        start_y = max(0, center_y - h_large // 2)
        end_x = min(img.shape[1], start_x + w_large)
        end_y = min(img.shape[0], start_y + h_large)

        # 확대된 객체가 삽입될 위치 계산
        roi = img[start_y:end_y, start_x:end_x]
        mask_roi = mask_large[:end_y-start_y, :end_x-start_x]
        mask_inv = cv2.bitwise_not(mask_roi)

        # ROI 영역에서 객체가 삽입될 부분을 검정색으로 만듦
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        obj_fg = cv2.bitwise_and(obj_large[:end_y-start_y, :end_x-start_x], 
                                 obj_large[:end_y-start_y, :end_x-start_x], 
                                 mask=mask_roi)

        # 원본 이미지에 객체 삽입
        combined = cv2.add(img_bg, obj_fg)
        img[start_y:end_y, start_x:end_x] = combined

    cv2.imshow("combined_image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
