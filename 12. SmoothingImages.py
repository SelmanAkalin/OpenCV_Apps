import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("opencv-logo.png")
assert img is not None, "file could not be read, check with os.path.exist()"

kernel = np.ones((5,5), np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title("Original")
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title("Averaging")
plt.xticks([]),plt.yticks([])
plt.show()

# Averaging
#blur = cv.blur(img,(5,5))

# Gaussian Blur. Highly effective against Gaussian filter.
#blur = cv.GaussianBlur(img,(5,5),0)

# Median. Highly effective against salt and pepper noise.
#blur = cv.medianBlur(img,5)

# Bilateral. Highly effective in noise removal while keeping edges sharp.
blur = cv.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()