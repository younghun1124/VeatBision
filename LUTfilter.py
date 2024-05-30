import numpy as np
import cv2

import numpy as np

def load_cube_lut(file_path):
    """CUBE 형식의 LUT 파일을 로드하여 numpy 배열로 반환합니다."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    lut_data = []
    size = 0

    for line in lines:
        if line.startswith('#') or line.startswith('TITLE'):
            continue
        elif line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
        else:
            values = list(map(float, line.strip().split()))
            if len(values) == 3:
                lut_data.append(values)
    
    lut_data = np.array(lut_data)
    expected_size = size ** 3    
    if len(lut_data) != expected_size:        
        raise ValueError(f"Invalid LUT file or size. Expected {expected_size} entries, got {len(lut_data)}.")
    lut = lut_data.reshape((size, size, size, 3))
    return lut

def apply_lut(image, lut):
    """LUT를 사용하여 이미지를 변환합니다."""
    lut_size = lut.shape[0]
    result = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j] / 255.0
            r_idx = int(r * (lut_size - 1))
            g_idx = int(g * (lut_size - 1))
            b_idx = int(b * (lut_size - 1))
            result[i, j] = lut[r_idx, g_idx, b_idx] * 255
    
    return result

def filter_image_with_lut(image, lut_path="luts\LUTS (2).cube"):
    """이미지와 LUT 파일을 받아 필터가 적용된 이미지를 반환합니다."""
    # 이미지 로드
   
    
    # LUT 파일 로드
    lut = load_cube_lut(lut_path)
    
    # LUT 적용
    filtered_image = apply_lut(image, lut)
    
    return filtered_image
