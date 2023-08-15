#https://pysource.com/2023/03/28/object-detection-with-yolo-v8-on-mac-m1-opencv-with-python-tutorial
import cv2
from ultralytics import YOLO
import numpy as np
import serial

serialcomm = serial.Serial("/dev/ttyS0", 9600)
cap = cv2.VideoCapture(0)
model = YOLO("best (1).pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cpu")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    cls: object
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        my_bytes=chr(cls).encode()
        serialcomm.write(my_bytes)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
