import time
import cv2
from angle_detect import contour_detect
start = time.time()
#testImg = '/home/toke/Desktop/image-test/2.jpg'
testImg = '/home/toke/Desktop/erode_with_dilate/2.png'
#testImg = '/home/toke/Desktop/erode_with_dilate/1.jpeg'
#testImg = '/home/toke/Desktop/print/color_crop.png'
img = cv2.imread(testImg)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY= cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient1 = cv2.convertScaleAbs(gradient)

blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)
#(_, thresh) = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY) #按某个阈值画二值图


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)    #close=先腐蚀后膨胀

# closed1 = cv2.erode(closed, None, iterations=4)
closed2 = cv2.dilate(closed, None, iterations=2)


# cv2.imshow('gray',gray)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gray).png',gray)

# cv2.imshow('gradientX',gradX)
# cv2.waitKey(0)
# cv2.imshow('gradientY',gradY)
# cv2.waitKey(0)

# cv2.imshow('gradient',gradient)

# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gradient.png',gradient)

# cv2.imshow('gradient1',gradient1)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gradient1.png',gradient1)

# cv2.imshow('blurred',blurred)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/blurred.png',blurred)

# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/thresh.png',thresh)

# cv2.imshow('closed',closed)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed.png',closed)

# cv2.imshow('closed1',closed1)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed1.png',closed1)

# cv2.imshow('closed2',closed2)
# cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed2.png',closed2)

img = '/home/toke/Desktop/erode_with_dilate/closed2.png'
contour_detect(img)

end = time.time()
print("Running time: %s seconds"%(end - start))