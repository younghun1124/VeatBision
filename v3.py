import cv2
import numpy as np
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def overlay_mask(image, mask, color, alpha=0.5):
    """마스크를 이미지 위에 오버레이합니다."""
    # 마스크를 RGB로 변환합니다.
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 마스크를 색상으로 채웁니다.
    mask = cv2.addWeighted(image, 1, mask, alpha, 0)
    return mask
    
# YOLOv8 모델을 로드합니다.
model = YOLO('yolov8n-seg.pt')

# 웹캠을 시작합니다.
cap = cv2.VideoCapture("data/2165-155327596_small.mp4")



while True:
    success, img = cap.read()
    if not success:
        break

    annotator = Annotator(img, line_width=2)

    results = model.track(img, persist=True)
   
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True))



    cv2.imshow("instance-segmentation-object-tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.waitKey(0)
    
cap.release()
cv2.destroyAllWindows()