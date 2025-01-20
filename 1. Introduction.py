import cv2 as cv
import sys

img = cv.imread("./cat.png")

if img is None:
    sys.exit("Couldn' t read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("c"):
    cv.imwrite("cat.png", img)