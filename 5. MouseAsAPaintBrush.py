import cv2 as cv
import numpy as np
# Here are all the mouse events
#events = [i for i in dir(cv) if "EVENT" in i]
#print(events)

drawing = False
mode = True
ix, iy, = -1, -1

# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            #else:
                #cv.circle(img, (ix, iy), (int(np.sqrt((np.square(ix)-np.square(x))+(np.square(iy)-np.square(y))))),(0, 0, 255), -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5,(0, 0 ,255), -1)

img = np.zeros((512,512,3), np.uint8)
cv.namedWindow("image")
cv.setMouseCallback("image", draw_circle)

while(1):
    cv.imshow("image", img)
    k = cv.waitKey(1)& 0xFF
    if k == ord("m"):
        mode = not mode
    elif k == 27:
        break
cv.destroyAllWindows()