from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

img2 = cv.imread('messi.png')
assert img2 is not None, "file could not be read, check with os.path.exists()"
rows,cols,ch = img2.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(img2,M,(300,300))
# Perspective Transformation
plt.subplot(121),plt.imshow(img2),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
# Scaling
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()