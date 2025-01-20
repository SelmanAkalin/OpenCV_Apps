# It is a multi-stage algorithm and we will go through each stages.
# Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter.
# We have already seen this in previous chapters
# Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction to get first derivative in horizontal direction and vertical direction.
# From these two images, we can find edge gradient and direction for each pixel
# After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted pixels which may not constitute the edge.
# For this, at every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient
# This stage decides which are all edges are really edges and which are not.
# For this, we need two threshold values, minVal and maxVal.
# Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded.
# Those who lie between these two thresholds are classified edges or non-edges based on their connectivity.
# If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded.

import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()