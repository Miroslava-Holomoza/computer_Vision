import os
import cv2
from ultralytics import YOLO
import time

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.5

VEHICLE_CLASSES = {
    2: 'Car',           
    3: 'Motorcycle',    
    5: 'Bus',         
    7: 'Truck',       
    1: 'Bicycle'      
}


CLASS_COLORS = {
    'Car': (0, 255, 0),          
    'Motorcycle': (255, 0, 0),    
    'Bus': (0, 165, 255),        
    'Truck': (0, 0, 255),         
    'Bicycle': (255, 255, 0)      
}

video_path = os.path.join(PROJECT_DIR, "transport.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Помилка: не можу відкрити відео {video_path}")
    exit()

fps_original = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Розміри відео: {width}x{height}")
print(f"FPS: {fps_original}")
print(f"Всього кадрів: {total_frames}")
print(f"Поріг впевненості: {CONF_THRESHOLD}")

RESIZE_WIDTH = 960 

prev_time = time.time()
fps = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

   
    result = model(frame, conf=CONF_THRESHOLD)[0]
    
    
    vehicle_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    total_vehicles = 0

    
    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
           
            if cls not in VEHICLE_CLASSES:
                continue
            
            vehicle_name = VEHICLE_CLASSES[cls]
            vehicle_counts[vehicle_name] += 1
            total_vehicles += 1
            
           
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
           
            color = CLASS_COLORS.get(vehicle_name, (255, 255, 255))
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
          
            label = f"{vehicle_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    now = time.time()
    dt = now - prev_time
    prev_time = now

    if dt > 0:
        fps = 1.0 / dt

 
    y_offset = 30
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    y_offset = 70
    for vehicle_name, count in vehicle_counts.items():
        if count > 0: 
            color = CLASS_COLORS.get(vehicle_name, (255, 255, 255))
            text = f"{vehicle_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 35
    
  
    cv2.putText(frame, f"Total vehicles: {total_vehicles}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
  
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
               (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.imshow("Traffic Detection - YOLO", frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Програма завершена!")
