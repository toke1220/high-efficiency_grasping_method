import cv2
import numpy as np
from PIL import Image, ImageEnhance
import math
from depth_KMeans import depth_data_KMeans
from depth_outliers import detect_outliers
import matplotlib.pyplot as plt
import copy
#testImg = '/home/toke/Desktop/erode_with_dilate/2.jpg'
#testImg = '/home/toke/Desktop/erode_with_dilate/closed2.png'
testImg = '/home/toke/Desktop/hough/2.png'   #10度
img = cv2.imread(testImg)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gray).png',gray)

cv2.imshow('gradient',gradient)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gradient.png',gradient)

cv2.imshow('gradient1',gradient1)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/gradient1.png',gradient1)

cv2.imshow('blurred',blurred)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/blurred.png',blurred)

cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/thresh.png',thresh)

cv2.imshow('closed',closed)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed.png',closed)

# cv2.imshow('closed1',closed1)
# cv2.waitKey(0)
# cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed1.png',closed1)

cv2.imshow('closed2',closed2)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed2.png',closed2)


img = '/home/toke/Desktop/erode_with_dilate/closed2.png'
img = cv2.imread(testImg)
binary = cv2.Canny(img, 30, 200)
cv2.imshow('canny',binary)
cv2.waitKey(0)
cv2.imwrite('/home/toke/Desktop/erode_with_dilate/closed2.png',closed2)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
# closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
# binary = cv2.Canny(closed, 30, 200)

#closed = cv2.erode(binary, None, iterations=1)
# cv2.imshow('closed', binary)
# cv2.waitKey(0)
# cv2.imshow('closed', closed)
# cv2.waitKey(0)
lines1 = cv2.HoughLines(binary, 1, np.pi/180, 200)
lines2 = cv2.HoughLinesP(binary, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)#统计概率霍夫线变换函数：图像矩阵，极坐标两个参数，一条直线所需最少的曲线交点，组成一条直线的最少点的数量，被认为在一条直线上的亮点的最大距离

ang = []
for line in lines2:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255),2)
        y1 = -y1
        y2 = -y2 - y1
        x2 = x2 - x1
        l = np.sqrt(x2**2 + y2**2)
        deg = math.asin(y2 / l)
        deg = 180 / math.pi * deg
        ang.append(deg)


angs = np.vstack(ang)
angs = angs.reshape(-1,1)

ang_clean = detect_outliers(angs, 1.5)
ang_clean = np.vstack(ang_clean)
ang_clean = ang_clean.reshape(-1,1)

ang_clean1 = copy.deepcopy(ang_clean)
for i in range(2):
    ang_clean1 = detect_outliers(ang_clean1, 0.1)
    ang_clean1 = np.vstack(ang_clean1)
    ang_clean1 = ang_clean1.reshape(-1,1)

ang1= np.average(ang_clean)
print(ang1)
ang2= np.average(ang_clean1)
print(ang2)

# depth_data_KMeans(angs, 3)
# depth_data_KMeans(ang_clean, 3)
# depth_data_KMeans(ang_clean1, 3)
# lines2[1][1] = -lines2[1][1]
# lines2[0][1] = -lines2[0][1]
# box1 = box[1]-box[0]
# f5 = np.sqrt(np.sum(box1**2))
# deg = math.asin(box1[1] /f5)
# deg = 180 / math.pi * deg

labels=['Depth_data']
flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}
# plt.boxplot(angs, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
# plt.show()
# plt.boxplot(ang_clean, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
# plt.show()

# plt.boxplot(ang_clean, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=0.1)
# plt.show()
# plt.boxplot(ang_clean1, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=0.1)
# plt.show()

# plt.boxplot(ang_clean1, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=0.1)
# plt.show()
# plt.boxplot(ang_clean1, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=0.1)
# plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img, 'RGB')
img.save('/home/toke/Desktop/hough/img_hough1.png')
img.show()
cv2.imshow('lines',binary)
cv2.waitKey(0)
cv2.imshow('lines',binary)
cv2.waitKey(0)