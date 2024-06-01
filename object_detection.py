import cv2
import numpy as np
import math
from ultralytics import YOLO
def size_changer(img, results, beat_coef):
    masks = np.zeros_like(img)
    boxes = np.zeros_like(img)
    # mask, class label 추출
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
    else:
        return img
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
    else:
        return img

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
        FX=1.0+beat_coef * 0.2
        FY=1.0+beat_coef * 0.2
        
        # 객체 크기를 두 배로 확대
        obj_large = cv2.resize(obj_cropped, None, fx=FX, fy=FY, interpolation=cv2.INTER_LINEAR)
        mask_large = cv2.resize(mask_cropped, None, fx=FX, fy=FY, interpolation=cv2.INTER_LINEAR)

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
        
    return img

def colorChange(img, results, beat, key,color_index=0):
    # 초기 확대 비율 설정
    scale_factor = 1.0

    # Generate a random color for each class
    num_classes = 100  # Assuming COCO dataset
    np.random.seed(0)
    colors = np.random.randint(0, 255, (num_classes, 3))
    masks = np.zeros_like(img)
    # mask, class label 추출
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
    else:
        return img
    # Create an empty image for masks
    mask_image = np.zeros_like(img)
        # Apply color to each mask
    for i in range(len(masks)):
        mask = masks[i]
        color = colors[color_index]
        
        # Resize the mask to match the size of mask_image
        mask = cv2.resize(mask, (mask_image.shape[1], mask_image.shape[0]))

        # Apply color to the mask
        mask_image[mask == 1] = color

    
    # Keep a copy of the original mask image
    original_mask_image = mask_image.copy()

    # Handle key inputs for scaling
    if key == ord(' '):
        cv2.waitKey()
    elif key == ord('a'):
        scale_factor *= 1.5
    elif key == ord('d'):
        scale_factor /= 1.5
    elif key == ord('[') or key == ord('{') or beat >=0.5:
        color_index = (color_index - 1) % num_classes
    elif key == ord(']') or key == ord('}') or beat < 0.5:
        color_index = (color_index + 1) % num_classes
    elif key == ord('z') or beat >=0.5:
        kernel = np.ones((15, 15), np.uint8)
        mask_image = cv2.dilate(original_mask_image, kernel, iterations=3)
    elif key == ord('c') or beat >=0.5:
        kernel = np.ones((5, 5), np.uint8)
        original_mask_image = original_mask_image.astype(np.uint8)
        mask_image = cv2.erode(original_mask_image, kernel, iterations=20)



    # Combine the original image with the mask image
    combined_image = cv2.addWeighted(img, 0.7, mask_image, 0.3, 0)

    # Resize mask image according to scale factor
    if scale_factor != 1.0:
        new_size = (int(mask_image.shape[1] * scale_factor), int(mask_image.shape[0] * scale_factor))
        mask_image_resized = cv2.resize(mask_image, new_size)
        img_resized = cv2.resize(img, new_size)  # Resize the original image
    else:
        mask_image_resized = mask_image
        img_resized = img  # No need to resize

    # Combine the original image with the mask image
    alhpa=beat
    combined_image = cv2.addWeighted(img_resized, 0.9, mask_image_resized, min(0.3, alhpa), 0)
    return combined_image