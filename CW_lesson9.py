import cv2

net = cv2.dnn.readNetFromCaffe("Data/MobileNet/mobilenet_deploy.prototxt", "Data/MobileNet/mobilenet.caffemodel") #завантаження моделі

classes = []
with open("Data/MobileNet/synset.txt", "r", encoding="utf-8") as f: #зчитуємо список
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread("images/MobileNet/cat.jpg")

blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))
#адаптуємо зображення під моднль

net.setInput(blob) #вкладення в мережу підготовленнш файлів

preds = net.forward() #вектор імовірності для класів
