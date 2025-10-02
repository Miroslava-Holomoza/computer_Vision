import cv2
import numpy as np
img = np.zeros((400,600,3), np.uint8)
img[:] = 142, 209, 155
me = cv2.imread('me.jpg')
me = cv2.resize(me, (120, 160))
x, y = 30, 30
h, w = me.shape[:2]
img[y:y+h, x:x+w] = me
cv2.rectangle(img, (10, 10), (590, 390), (29, 66, 36), 3)
cv2.putText(img, "Miroslava Holomoza", (180, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9,(29, 66, 36))
cv2.putText(img, "Computer vision student", (180, 140), cv2.FONT_HERSHEY_DUPLEX, 0.9,(29, 66, 36))
cv2.putText(img, "Email: miroslava.holomoza@gmail.com", (180, 210), cv2.FONT_HERSHEY_DUPLEX, 0.5,(29, 66, 36))
cv2.putText(img, "Phone: 066 645 0014", (180, 250), cv2.FONT_HERSHEY_DUPLEX, 0.5,(29, 66, 36))
cv2.putText(img, "16/11/2009", (180, 290), cv2.FONT_HERSHEY_DUPLEX, 0.5,(29, 66, 36))
cv2.putText(img, " OpenCV Business Card ", (150, 350), cv2.FONT_HERSHEY_DUPLEX, 0.7,(16, 18, 16))

qr = cv2.imread('frame.png')
qr = cv2.resize(qr, (120, 120))
x1, y1 = 450, 250  
h1, w1 = qr.shape[:2]
img[y1:y1+h1, x1:x1+w1] = qr

cv2.imshow('Image', img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
