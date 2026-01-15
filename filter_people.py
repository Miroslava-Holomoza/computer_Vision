import cv2
import numpy as np
import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

cascade_path = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error loading cascade classifier")
    exit()

def detect_people(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

files = os.listdir(IMAGES_DIR)
counter_people = 0
counter_no_people = 0

for filename in files:
    for ext in allowed_extensions:
        if filename.lower().endswith(ext):
            break
    else:
        continue

    img_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue

    faces = detect_people(img)
    if len(faces) > 0:
        out_path = os.path.join(PEOPLE_DIR, filename)
        shutil.copyfile(img_path, out_path)
        counter_people += 1

        boxed = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        boxed_path = os.path.join(PEOPLE_DIR, f"boxed_{filename}")
        cv2.imwrite(boxed_path, boxed)

    else:
        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copyfile(img_path, out_path)

        counter_no_people += 1

print(f"Total images with people: {counter_people}")
print(f"Total images without people: {counter_no_people}")