import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):
    
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    gry = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gry,100,200)
    # define range of blue color in HSV
    #mask = cv.inRange(gry, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=edges)

    cv.imshow("frame", frame)
    cv.imshow("mask", edges)
    cv.imshow("res", res)
    k = cv.waitKey(5) & 0XFF
    if k == 27:
        break

img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)

cv.destroyAllWindows()