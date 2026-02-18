import os
import sys
import cv2
from ultralytics import YOLO
import time
import subprocess
import numpy as np
from collections import defaultdict
import json

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

# YouTube URL - прямий ефір
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

print(f"⬇️  Отримання потоку з YouTube: {YOUTUBE_URL}")
print(f"   Це потрібно один раз для отримання прямого URL...")

# Отримати прямий URL потоку через yt-dlp
try:
    result = subprocess.run(
        ["yt-dlp", "-f", "best", "-g", YOUTUBE_URL],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        stream_url = result.stdout.strip().split('\n')[0]  # Беремо перший URL
        print(f"✓ Потік отримано успішно!")
        print(f"   Підключення до потоку...")
    else:
        print(f"❌ Помилка отримання потоку: {result.stderr}")
        exit(1)
        
except Exception as e:
    print(f"❌ Помилка: {e}")
    print(f"   Переконайтеся, що yt-dlp встановлено: pip install yt-dlp")
    exit(1)

# Завантажити YOLO модель
print("Завантаження YOLO моделі...")
try:
    model = YOLO("yolov8n.pt")
    print("YOLO модель завантажена успішно")
except Exception as e:
    print(f"Помилка завантаження YOLO моделі: {e}")
    exit(1)

CONF_THRESHOLD = 0.5

# Класи машин в YOLO
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

# Відкрити відео потік
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"❌ Помилка: не можу підключитися до потоку")
    exit()

print(f"✓ Підключено до потоку!")
print(f"\n🎥 Запуск детекції машин у реальному часі...")
print(f"   Натиснете 'q' у вікні для завершення\n")

RESIZE_WIDTH = 960

# Змінні для отслідування об'єктів та швидкості
prev_time = time.time()
fps = 0.0
frame_count = 0
prev_centroids = {}  # {object_id: (x, y)}
car_speeds = []  # Список всіх виявлених швидкостей машин
object_id_counter = 0
distance_threshold = 100  # Пікселі для зв'язання об'єктів між кадрами

# Глобальний лічильник машин
total_cars_detected = 0
frame_height = None
frame_width = None

def calculate_distance(p1, p2):
    """Обчислити евклідову відстань між двома точками"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def match_detections(prev_centroids, current_boxes, distance_threshold):
    """Зв'язати поточні детекції з попередніми"""
    matched = {}
    unmatched_current = list(range(len(current_boxes)))
    used_indices = set()
    
    for obj_id, prev_centroid in prev_centroids.items():
        min_distance = float('inf')
        min_idx = -1
        
        for i, box in enumerate(current_boxes):
            if i in used_indices:
                continue
            x1, y1, x2, y2 = box
            current_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = calculate_distance(prev_centroid, current_centroid)
            
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                min_idx = i
        
        if min_idx != -1:
            matched[obj_id] = current_boxes[min_idx]
            used_indices.add(min_idx)
    
    unmatched_current = [i for i in range(len(current_boxes)) if i not in used_indices]
    return matched, unmatched_current

# Піксель-до-метра коефіцієнт (приблизно для дорожних камер)
PIXEL_TO_METER = 0.01  # 1 піксель = 1 см (можна налаштувати)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Визначити розміри кадру при першому запуску
    if frame_height is None:
        frame_height, frame_width = frame.shape[:2]
        print(f"✓ Розміри кадру: {frame_width}x{frame_height}")
    
    # Змінити розмір кадру для швидкості обробки
    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    # Запустити YOLO детекцію
    result = model(frame, conf=CONF_THRESHOLD)[0]
    
    # Зібрати всі поточні детекції машин
    current_vehicle_boxes = []
    vehicle_info = []
    
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
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            current_vehicle_boxes.append((x1, y1, x2, y2))
            vehicle_info.append({
                'box': (x1, y1, x2, y2),
                'vehicle_name': vehicle_name,
                'conf': conf,
                'cls': cls
            })
    
    # Зв'язати об'єкти з попередніх кадрів
    matched_objects, unmatched_indices = match_detections(
        prev_centroids, current_vehicle_boxes, distance_threshold
    )
    
    # Оновити ID для нових об'єктів
    new_prev_centroids = {}
    speeds_this_frame = []
    
    for obj_id, box in matched_objects.items():
        x1, y1, x2, y2 = box
        current_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        prev_centroid = prev_centroids[obj_id]
        
        # Обчислити переміщення в пікселях
        pixel_distance = calculate_distance(prev_centroid, current_centroid)
        
        # Конвертувати в метри
        meter_distance = pixel_distance * PIXEL_TO_METER
        
        # Обчислити швидкість (м/сек)
        if fps > 0:
            time_delta = 1.0 / fps
            speed = meter_distance / time_delta  # м/сек
            speed_kmh = speed * 3.6  # конвертувати в км/год
            
            if speed_kmh > 0:  # Тільки записувати помітні рухи
                car_speeds.append(speed_kmh)
                speeds_this_frame.append(speed_kmh)
        
        new_prev_centroids[obj_id] = current_centroid
    
    # Додати нові об'єкти
    for idx in unmatched_indices:
        x1, y1, x2, y2 = current_vehicle_boxes[idx]
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        new_prev_centroids[object_id_counter] = centroid
        total_cars_detected += 1  # Рахуємо тільки нові унікальні машини
        object_id_counter += 1
    
    prev_centroids = new_prev_centroids
    
    # Намалювати детекції на кадрі
    vehicle_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    total_vehicles = 0
    
    for info in vehicle_info:
        x1, y1, x2, y2 = info['box']
        vehicle_name = info['vehicle_name']
        conf = info['conf']
        
        vehicle_counts[vehicle_name] += 1
        total_vehicles += 1
        
        color = CLASS_COLORS.get(vehicle_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{vehicle_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
    # Обчислити FPS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    
    if dt > 0:
        fps = 1.0 / dt
    
    # Вивести статистику на кадр
    # Вивести глобальну кількість машин
    y_offset = 30
    cv2.putText(frame, f"Total Vehicles: {total_cars_detected}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Вивести середню швидкість
    if car_speeds:
        avg_speed = np.mean(car_speeds)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} km/h", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    # Показати кадр
    try:
        cv2.imshow("Traffic Detection - YOLO with Speed", frame)
        # Натиснути 'q' для виходу
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Помилка при показі вікна: {e}")
        break

# Завершити обробку
cap.release()
cv2.destroyAllWindows()

# Вивести фінальну статистику
print("\n" + "="*60)
print("ФІНАЛЬНА СТАТИСТИКА ДЕТЕКЦІЇ")
print("="*60)
print(f"Всього оброблено кадрів: {frame_count}")
print(f"✓ Всього виявлено машин: {total_cars_detected}")
if car_speeds:
    print(f"\nШвидкість:")
    print(f"  - Середня: {np.mean(car_speeds):.2f} км/год")
    print(f"  - Максимальна: {np.max(car_speeds):.2f} км/год")
    print(f"  - Мінімальна: {np.min(car_speeds):.2f} км/год")
    print(f"  - Кількість виявлених рухів: {len(car_speeds)}")
else:
    print("⚠️  Недостатньо даних для обчислення швидкості")
print("="*60)
print("✓ Програма завершена!")
