import cv2
import numpy as np

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]#масштабування координат облицччя
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0)) #багатовимірний масив із зображеннями готової моделі
    face_net.setInput(blob)
    detections = face_net.forward() #отримання виявлених облич
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0,startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 1)
            
            roi_gray = cv2.cvtColor(frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)
            roi_color = frame[startY:endY, startX:endX]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)



    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
