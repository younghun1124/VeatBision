import cv2
import numpy as np
from music_extract import extract_beat_timing
from LUTfilter import filter_image_with_lut

# 비디오 캡처

cap = cv2.VideoCapture("data/2165-155327596_small.mp4")


# musicpath=''
# extract_beat_timing(musicpath)
while True:
    # Read an image from 'video'
    valid, img = cap.read()
    img = filter_image_with_lut(img)  # Apply LUT filter to the image
    if not valid:
        break    
    # Show the image
    cv2.imshow('Video Player', img)

    # Terminate if the given key is ESC
    key = cv2.waitKey()
    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()