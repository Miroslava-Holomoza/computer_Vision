import cv2
import numpy as np

img = cv2.imread('images/men.jpg')
scale = 2
img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
#make a copy
img_copy = img.copy()
img_copy_color = img_copy.copy()

cv2.imshow('men', img)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
#blur
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)
#contrast
img_copy = cv2.equalizeHist(img_copy)
#make the borders
img_copy = cv2.Canny(img_copy, 100, 150)
#find the borders
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#апроксимація - процес вираження одних величин або об'єктів через інші

#малювання контурів прямокутників до тексту
for cnt in contours:
    area = cv2.contourArea(cnt)#contours S
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)#make the smallest rectangle
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)#draw contours
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_y = y - 5 if  y - 5 > 10 else y + 15
        text = f"x: {x}, y: {y}, S:{int(area)}"
        cv2.putText(img_copy_color, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('men copy', img_copy_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
