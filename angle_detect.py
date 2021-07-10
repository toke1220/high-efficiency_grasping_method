# coding=UTF-8
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import copy


def contour_detect(testImg):
    img = cv2.imread(testImg)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # canny边缘检测
    binary = cv2.Canny(img_gray, 30, 200)
    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    #binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    canny_result = cv2.drawContours(img, contours, -1, (0,255,0), 2)
    # cv2.imwrite('/home/toke/Desktop/image/canny_result.png',canny_result)     #保存canny边缘检测结果
    # cv2.imwrite('/home/toke/Desktop/image/canny_binary.png',binary) #保存二值图

    edge_points = []
    for points in contours:
        for point in points:
            if 0.15*len(img)<point[0][1]<0.85*len(img) and 0.15*np.shape(img)[1]<point[0][0]<0.85*np.shape(img)[1]: #0.85full image内的边缘点
            #if (point[0][1] < 0.5*len(img) and point[0][0]<0.5*np.shape(img)[1]) or (point[0][1] > 0.5*len(img) and point[0][0]<0.5*np.shape(img)[1]) or (point[0][1] < 0.5*len(img) and point[0][0]>0.5*np.shape(img)[1]) or (point[0][1] > 0.5*len(img) and point[0][0]>0.5*np.shape(img)[1]):
                edge_points.append(point.reshape(1,2))
    edge_points_m = np.vstack(edge_points)
    for e in edge_points_m:
        cv2.circle(img, e, 1, (0, 0, 255), -1)
    cv2.imwrite('/home/toke/Desktop/image/test.png',img)    

    rect = cv2.minAreaRect(edge_points_m) #生成最小内接矩阵
    print(rect)

    box = cv2.boxPoints(rect)  #点顺时针生成，注：0,1两点不一定是最长边
    box = np.int0(box)
    
    # i=5               #画内接矩阵四个顶点
    # for p in box:
    #     i += 2
    #     cv2.circle(img, p, i, (0, 0, 255), -1)

    cv2.polylines(img,[box],True,(0,255,255),3)  #画内接矩阵
    #cv2.imwrite('/home/toke/Desktop/image/color_crop_rect.png',img)


    #取0,1对和0,3对
    f1 = np.sqrt(np.sum((box[0] - box[1])**2))
    f2 = np.sqrt(np.sum((box[0] - box[3])**2))
    if f1 >= f2:
        box[1][1] = -box[1][1]
        box[0][1] = -box[0][1]
        box1 = box[1]-box[0]
        f5 = np.sqrt(np.sum(box1**2))
        deg = math.asin(box1[1] /f5)
        deg = 180 / math.pi * deg
    else:
        box[3][1] = -box[3][1]
        box[0][1] = -box[0][1]
        box1 = box[3]-box[0]
        f5 = np.sqrt(np.sum(box1**2))
        deg = math.asin(box1[1] / f5)
        deg = 180 / math.pi * deg
    print('wxm_deg = {}'.format(deg))
    print('rect_a_deg = {}'.format(rect[2]))

if __name__ == "__main__":

    #img = '/home/toke/Desktop/image/color_crop.png'
    img = '/home/toke/Desktop/image/closed2.png'
    contour_detect(img)

