import cv2 as cv
import numpy as np
"""
def renk_bilgisi(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Fare tıklaması olduğunda pikselin renk değerlerini al
        renk = img[y, x]
        b, g, r = renk
        print(f"BGR: ({b}, {g}, {r})")

# Fotoğrafı oku
img = cv.imread('kamera_fotografi1.jpg')

# Pencereye tıklama olayını bağla
cv.namedWindow('Fotoğraf')
cv.setMouseCallback('Fotoğraf', renk_bilgisi)

# Fotoğrafı göster
cv.imshow('Fotoğraf', img)

# Kullanıcı bir tuşa basana kadar bekle
cv.waitKey(0)

# Pencereyi kapat
cv.destroyAllWindows()
"""
"""
# Kamera bağlantısını aç
cap = cv.VideoCapture(0)

# Kamera açıldıysa
if cap.isOpened():
    # Kameradan bir kare oku
    ret, frame = cap.read()
    
    if ret:
        # Çekilen kareyi göster
        cv.imshow('Kamera', frame)

        # Çekilen kareyi kaydet
        cv.imwrite('kamera_fotografi1.jpg', frame)

        # Kullanıcı bir tuşa basana kadar bekle
        cv.waitKey(0)

# Kamerayı serbest bırak
cap.release()

# Pencereyi kapat
cv.destroyAllWindows()
"""

cap = cv.VideoCapture(0)

while(1):
    
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([60, 60, 30])
    upper_blue = np.array([200, 200, 160])
    # define range of skin color in HSV
    #lower_blue = np.array([0, 33, 65])
    #upper_blue = np.array([20, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)
    k = cv.waitKey(5) & 0XFF
    if k == 27:
        break

# How to find HSV values to track?
#green = np.uint8([[[0,255,0]]])
#hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
#print(hsv_green)

cv.destroyAllWindows()