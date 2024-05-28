import cv2
import numpy as np
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def overlay_mask(results, img):
    for r in results:
        img = np.copy(r.orig_img)

        # Iterate each object contour 
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]

            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Create contour mask 
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Choose one:

            # OPTION-1: Isolate object with black background
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)

            # # OPTION-2: Isolate object with transparent background (when saved as PNG)
            # isolated = np.dstack([img, b_mask])

            # OPTIONAL: detection crop (from either OPT1 or OPT2)
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]

    return isolated
    
# YOLOv8 모델을 로드합니다.
model = YOLO('yolov8n-seg.pt')

# 웹캠을 시작합니다.
cap = cv2.VideoCapture("data/2165-155327596_small.mp4")



while True:
    success, img = cap.read()
    if not success:
        break

    annotator = Annotator(img, line_width=2)

    results = model.predict(img)
   
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True))

    # 원본 이미지 위에 마스크 오버레이
    image_with_masks = overlay_mask(results, img)


    cv2.imshow("instance-segmentation-object-tracking", image_with_masks)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.waitKey(0)
    
cap.release()
cv2.destroyAllWindows()