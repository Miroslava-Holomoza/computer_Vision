import cv2
import numpy as np

img = cv2.imread("image.jpg")
img = cv2.resize(img, (640, 480))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

lower_green = np.array([40, 50, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

mask_total = cv2.bitwise_or(mask_red, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_blue)

kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 800:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img, [cnt], -1, (62, 27, 99), 2)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # центр мас
            cv2.circle(img, (cx, cy), 4, (0, 255, 255), 2)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 4:
            shape = "square"
        elif len(approx) == 3:
            shape = "triangle"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "other"

        mask_object = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_object, [cnt], -1, 255, -1)
        mean_val = cv2.mean(hsv, mask=mask_object)[:3]
        h_val = mean_val[0]

        if (0 <= h_val <= 10) or (170 <= h_val <= 179):
            color = "red"
        elif 40 <= h_val <= 85:
            color = "green"
        elif 90 <= h_val <= 130:
            color = "blue"
        else:
            color = "unknown"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(img, f"Area: {int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Perimeter: {int(perimeter)}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, f'Shape:{shape}', (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(img, f'Color: {color}', (x - 10, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f"x: {x}, y: {y}"
        cv2.putText(img, text, (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.imshow("Mask Red", mask_red)
cv2.imshow("Mask Green", mask_green)
cv2.imshow("Mask Blue", mask_blue)
cv2.imshow("All Objects", img)
cv2.imwrite("result.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

