import os
import cv2
from ultralytics import YOLO
import time

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    video_path = os.path.join(PROJECT_DIR, "video_with_animals.mp4")
    cap = cv2.VideoCapture(video_path)

model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.4

RESIZE_WIDTH = 960 

prev_time = time.time()
fps = 0.0


CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf=CONF_THRESHOLD)[0]
    animals_count = 0
    psevdo_id = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == CAT_CLASS_ID or cls == DOG_CLASS_ID:
                animals_count += 1
                psevdo_id += 1
                animal_type = "Cat" if cls == CAT_CLASS_ID else "Dog"
                label = f"ID:{psevdo_id} {animal_type} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Animals count: {animals_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break