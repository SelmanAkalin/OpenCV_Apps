import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
#  It is useful for removing small white noises , detach two connected objects etc.
erosion = cv.erode(img,kernel,iterations = 1)
# It is just opposite of erosion.  It is also useful in joining broken parts of an object.
dilation = cv.dilate(img,kernel,iterations = 1)
# Opening is just another name of erosion followed by dilation. It is useful in removing noise
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# It is the difference between dilation and erosion of an image.
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
# It is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel.
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# It is the difference between the closing of the input image and input image.
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

plt.subplot(331),plt.imshow(img),plt.title("Original")
plt.subplot(332),plt.imshow(erosion),plt.title("Erosion")
plt.subplot(333),plt.imshow(dilation),plt.title("Dilation")
plt.subplot(334),plt.imshow(opening),plt.title("Opening")
plt.subplot(335),plt.imshow(closing),plt.title("Closing")
plt.subplot(336),plt.imshow(gradient),plt.title("Morphological Gradient")
plt.subplot(337),plt.imshow(tophat),plt.title("TopHat")
plt.subplot(338),plt.imshow(blackhat),plt.title("BlackHat")

plt.show()