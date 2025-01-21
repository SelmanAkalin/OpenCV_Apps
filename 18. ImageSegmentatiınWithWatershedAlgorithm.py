import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_original = cv.imread("coins.png")
assert img_original is not None, "file could not be read, check with os.path.exists()"
img = img_original.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations=3)
opening = cv.morphologyEx(closing,cv.MORPH_OPEN,kernel,iterations=1)

# sure background area
sure_bg = cv.dilate(closing,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(closing,cv.DIST_L2,5)
_, sure_fg = cv.threshold(dist_transform,0.6*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,255]

plt.subplot(331),plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB)),plt.title("Original")
plt.subplot(332),plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB)),plt.title("Gray")
plt.subplot(333),plt.imshow(cv.cvtColor(thresh, cv.COLOR_BGR2RGB)),plt.title("Thresh")
plt.subplot(334),plt.imshow(cv.cvtColor(opening, cv.COLOR_BGR2RGB)),plt.title("Opening")
plt.subplot(335),plt.imshow(cv.cvtColor(closing, cv.COLOR_BGR2RGB)),plt.title("Closing")
plt.subplot(336),plt.imshow(cv.cvtColor(sure_bg, cv.COLOR_BGR2RGB)),plt.title("Sure background")
plt.subplot(337),plt.imshow(cv.cvtColor(sure_fg, cv.COLOR_BGR2RGB)),plt.title("Sure foreground")
plt.subplot(338),plt.imshow(cv.cvtColor(unknown, cv.COLOR_BGR2RGB)),plt.title("Unknown")
plt.subplot(339),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title("Markers")
plt.subplots_adjust(hspace=0.5)
plt.show()
cv.waitKey(0)