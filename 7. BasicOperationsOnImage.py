import cv2 as cv
import numpy as np

img = cv.imread("messi.png")
assert img is not None, "file could not be read with os.path.exist()"

px = img[100, 100]
print(px)

blue = img[100, 100, 0]
print(blue)

temp = img[100, 100]
img[100, 100] = [255, 255, 255]
print(img[100, 100])
img[100, 100] = temp
print(img.shape)
print(img.size)
print(img.dtype)
x = img[213:273, 100:160]
ball = img[215:275, 260:320]
img[213:273, 100:160] = ball
cv.imshow("image", img)
cv.waitKey()
img[213:273, 100:160] = x

# split and merge channels
b, g, r = cv.split(img)
img = cv.merge((b,g,r))
# or split with numpy indexing
b = img[:,:,0]
# turning all reds into zero
#img[:,:,2] = 0