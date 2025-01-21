import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('star.png')
assert img is not None, "file could not be read, check with os.path.exists()"

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(img_gray,127,255,0)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Moments helps us calculate some features like center of mass of the object, area etc.
cnt = contours[0]

epsilon = 0.02*cv.arcLength(cnt,True)

# Tight Contour and Hull
img2 = img.copy()
approx = cv.approxPolyDP(cnt,epsilon,True)
cv.drawContours(img2, [approx], 0, (0, 255, 0), 2)
hull = cv.convexHull(cnt)
cv.drawContours(img2, [hull], 0, (255,0,0), 2)

# Bounding Rectangle
img3 = img.copy()
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),2)

# Rotated Rectangle
img4 = img.copy()
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int32(box)
cv.drawContours(img4,[box],0,(0,0,255),2)

# Detecting Defects
img5 = img.copy()
hull2 = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull2)
 
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img5,start,end,[0,255,0],2)
    cv.circle(img5,far,5,[255,0,0],-1)

# Matching Shapes
img_rotated = cv.imread('rotated_star.jpg', cv.IMREAD_GRAYSCALE)
_, thresh_r = cv.threshold(img_rotated, 127, 255, 0)
contours_r,_ = cv.findContours(thresh_r,2,1)
cnt_r = contours_r[0]
img_scaled = cv.imread('rotated_and_scaled_star.png', cv.IMREAD_GRAYSCALE)
_, thresh_s = cv.threshold(img_scaled, 127, 255, 0)
contours_s,_ = cv.findContours(thresh_s,2,1)
cnt_s = contours_s[0]

match_point1 = cv.matchShapes(cnt,cnt,1,0.0)
match_point2 = cv.matchShapes(cnt,cnt_r,1,0.0)
match_point3 = cv.matchShapes(cnt,cnt_s,1,0.0)
print("Match Point with itself", match_point1)
print("Match Point with rotated", match_point2)
print("Match Point with rotated and scaled", match_point3)

plt.subplot(321),plt.imshow(img2),plt.title("Tight Contour(G)/ Hull(B)")
plt.subplot(322),plt.imshow(img3),plt.title("Bounding Box")
plt.subplot(323),plt.imshow(img4),plt.title("Rotated Rectangle")
plt.subplot(324),plt.imshow(img5),plt.title("Defects")
plt.subplot(325),plt.imshow(img_rotated),plt.title("Rotated Image")
plt.subplot(326),plt.imshow(img_scaled),plt.title("Rotated and Scaled Image")
plt.subplots_adjust(hspace=0.5)
plt.show()
cv.waitKey(0)