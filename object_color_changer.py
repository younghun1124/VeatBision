import cv2
import numpy as np
import math
from ultralytics import YOLO



def colorChange(img, results, beat):
    # 초기 확대 비율 설정
    scale_factor = 1.0

    # Generate a random color for each class
    num_classes = 100  # Assuming COCO dataset
    np.random.seed(0)
    colors = np.random.randint(0, 255, (num_classes, 3))
    color_index = 0

    # mask, class label 추출
    masks= results[0].masks.data.cpu().numpy()
    
    classes = results[0].boxes.cls.cpu().numpy()

    
    # Create an empty image for masks
    mask_image = np.zeros_like(img)
        # Apply color to each mask
    for i in range(len(masks)):
        mask = masks[i]
        class_id =int ( classes[i])
        color = colors[color_index]
        
        # Resize the mask to match the size of mask_image
        mask = cv2.resize(mask, (mask_image.shape[1], mask_image.shape[0]))

        # Apply color to the mask
        mask_image[mask == 1] = color

    
    # Keep a copy of the original mask image
    original_mask_image = mask_image.copy()

    # Handle key inputs for scaling
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.waitKey()
    elif key == ord('a'):
        scale_factor *= 1.5
    elif key == ord('d'):
        scale_factor /= 1.5
    elif key == ord('[') or key == ord('{') or beat >=0.5:
        color_index = (color_index - 1) % num_classes
    elif key == ord(']') or key == ord('}') or beat >=0.5:
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
    combined_image = cv2.addWeighted(img_resized, 0.7, mask_image_resized, 0.3+alhpa, 0)
    return combined_image
    
