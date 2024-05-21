from ultralytics import YOLO
import cv2


model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened() :
    ret, frame = cap.read()
    wait_msec = int(1 / fps * 1000)
    if ret :
        results = model(frame)
        cv2.imshow("Results", results[0].plot())


    key=cv2.waitKey(wait_msec)
    if key == ord(' ') :
        key=cv2.waitKey()
    if key == 27 :
        break

cv2.destroyAllWindows()
cap.release()