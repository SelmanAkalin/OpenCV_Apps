import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("chessboard.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Most accurate 30 corners
corners = cv.goodFeaturesToTrack(gray,30,0.01,10)
corners = np.float32(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (int(x),int(y)),3,255,-1)

plt.subplot(1,1,1),plt.imshow(img),plt.title("Most accurate 30 corners.")
plt.show()