import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while(1):
    _, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(frame, (5,5),0)
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(blurred_frame, cv.MORPH_OPEN, kernel)
    laplacian = cv.Laplacian(blurred_frame,cv.CV_64F)
    canny1 = cv.Canny(opening,100,200)
    canny2 = cv.Canny(frame,100,200)
    #canny3 = cv.Canny(blurred_frame,100,150)
    #canny4 = cv.Canny(frame,100,150)

    cv.imshow("Frame", frame)
    cv.imshow("Canny1", canny1)
    cv.imshow("Canny2", canny2)
    #cv.imshow("Canny3", canny3)
    #cv.imshow("Canny4", canny4)

    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()