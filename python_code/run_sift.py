import numpy as np
import cv2 as cv
import time


img = cv.imread('/Users/aminaizbassar/Documents/cpp_proj/scan65/image/000000.png')
mask = cv.imread('/Users/aminaizbassar/Documents/cpp_proj/scan65/mask/000.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)


sift = cv.SIFT_create()
start = time.time()
kp, des = sift.detectAndCompute(gray, mask, None)
end = time.time() - start
print(end)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('sift_keypoints.jpg',img)