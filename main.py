# coding:utf-8
''' this code is used to acquire object coordinate from camera frame through geometry calculation, and add contour detection. completed on 20, July.'''

from PIL import Image, ImageFont, ImageDraw
from sklearn.cluster import KMeans
from scipy import ndimage
from angle_detect import contour_detect
from depth_outliers import detect_outliers
from depth_KMeans import depth_data_KMeans
from cluster_c import k_silhouette
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import glob as gb
import pyyolo
import cv2
import torch

import math
import time
import os
import darknet


# import sys                     #ros库冲突
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)

#####-------------------------------相机初始化-------------------------------#####

pipeline = rs.pipeline()  #接收相机信息
config = rs.config()      #相机信息配置
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)   #848*480
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   #848*480
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)   #打开相机
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()   #检索深度图的单位和meters单位之间的映射 (深度比例)

align_to = rs.stream.color
align = rs.align(align_to)  #执行深度图像与另一个图像之间的对其

#####---------------------------图像color和depth对齐-------------------------------#####
def get_aligned_imgs():
    for i in range(50):
        frames = pipeline.wait_for_frames()     #获取Realsense一帧的数据
        aligned_frames = align.process(frames)   #在给定的帧上运行对齐过程，以获得一组对齐的帧

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imwrite('/home/toke/Desktop/image/camera_catch.png', color_image)
        cv2.imwrite('/home/toke/Desktop/image/camera_catch_depth.png', depth_image)
    return color_image, depth_image
#color_image, depth_image = get_aligned_imgs()

def bbox_yolov4(testImg):
    ### python3 run yolov4 model
    img = Image.open(testImg)   #PIL库的图片读取函数 （W x H 不含通道数）
    fname = os.path.basename(testImg)
    draw = ImageDraw.Draw(img)  #在图像上绘制一些东西
    image = cv2.imread(testImg) 

    detector = pyyolo.YOLO("/home/toke/Packages/darknet/yolov4_laji/yolov4_garbage_origin.cfg",
                           "/home/toke/Packages/darknet/yolov4_laji/yolov4_garbage_origin_best.weights",
                           "/home/toke/Packages/darknet/yolov4_laji/garbage_0201.data",
                           detection_threshold = 0.5,
                           hier_threshold = 0.5,
                           nms_threshold = 0.45)   
    # detector = pyyolo.YOLO("/home/toke/Packages/darknet/test_models/yolov4-custom.cfg",
    #                        "/home/toke/Packages/darknet/test_models/yolov4.weights",
    #                        "/home/toke/Packages/darknet/test_models/coco.data",
    #                        detection_threshold = 0.5,
    #                        hier_threshold = 0.5,
    #                        nms_threshold = 0.45)   

    bboxes, center_points = [], []
    dets = detector.detect(image, rgb=False)  #返回det bbox（x, y, w, h） list object
    for i, det in enumerate(dets):   #会有多个bbox
        
        
        #print ('Detection:', {i}, {det})
        f1 = (dets[i][2] - 0.7*dets[i][2]) / 2
        f2 = (dets[i][3] - 0.7*dets[i][3]) / 2

        dets[i][2] = 0.7*dets[i][2] #w     框缩小
        dets[i][3] = 0.7*dets[i][3] #hds
        dets[i][0] = dets[i][0] + f1  #x
        dets[i][1] = dets[i][1] + f2  #y

        xmin, ymin, xmax, ymax = det.to_xyxy()  #将bbox转成四个点x
        bbox = [xmin, ymin, xmax, ymax]

        center_point = (math.ceil((xmin + xmax) / 2), math.ceil((ymin + ymax) / 2))  #bbox中心点
        
        if i < np.shape(dets)[0]-1:
            draw.ellipse((center_point[0], center_point[1], center_point[0] + 10, center_point[1] + 10), 'red') #画bbox的中心点
            draw.rectangle((bbox[0],bbox[1], bbox[2], bbox[3]),  outline='red', fill='black')
            
        bboxes = np.vstack(bbox)
        center_points = np.vstack(center_point)
        
    # draw.ellipse((center_point[0], center_point[1], center_point[0] + 10, center_point[1] + 10), 'hotpink')
    # draw.polygon([(bbox[0],bbox[1]),(bbox[0],bbox[3]),(bbox[0],bbox[1]),(bbox[2],bbox[1])], outline='mediumorchid')
    # draw.polygon([(bbox[2],bbox[3]),(bbox[0],bbox[3]),(bbox[2],bbox[3]),(bbox[2],bbox[1])], outline='mediumorchid')

    draw.ellipse((center_point[0], center_point[1], center_point[0] + 10, center_point[1] + 10), 'red')
    draw.polygon([(bbox[0],bbox[1]),(bbox[0],bbox[3]),(bbox[0],bbox[1]),(bbox[2],bbox[1])], outline='red')
    draw.polygon([(bbox[2],bbox[3]),(bbox[0],bbox[3]),(bbox[2],bbox[3]),(bbox[2],bbox[1])], outline='red')
    img.show()

    #path = '/home/toke/Desktop/image'
    img.save('/home/toke/Desktop/image/img_bbox.png')
    # fname = os.path.basename(testImg)
    # img.save(os.path.join(path, fname))

    return bboxes, center_points, bbox

#####---------------------------深度值聚类&画图-------------------------------#####
#def depth_data_KMeans(y,k):
    for k in range(2,k+1):
        clf = KMeans(n_clusters=k) #设定k  这里就是调用KMeans算法，同时默认参数就是KMeans++
        s = clf.fit(y) #加载数据集合
        numSamples = len(y) 
        centroids_label = clf.labels_
        print (centroids_label, type(centroids_label)) #显示各个点的，聚类标签
        #print (clf.inertia_)  #显示聚类效果 评价指标
        mark_data = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        #画出所有样例点 属于同一分类的绘制同样的颜色
        for i in range(numSamples):
            plt.plot(y[i][0], 0, mark_data[clf.labels_[i]]) #mark[markIndex])
        mark_centroids = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出质点，用特殊图型
        centroids =  clf.cluster_centers_
        for i in range(k):
            plt.plot(centroids[i][0], 0, mark_centroids[i], markersize = 12)
            print('><><><><><><><><><><><>')
            print(centroids[i][0])   #输出聚类中心值
            plt.axvline(centroids[i][0], ymin=0.3, ymax=0.7)
        plt.show()
    return 0

#####-------------------------------抓取定位,内参矩阵-------------------------------#####
def piexl_to_3d_point_hands(intr,point,depth_frame):   
    fx = intr['fx']
    fy = intr['fy']
    ppx = intr['ppx']
    ppy = intr['ppy']
    coeffs = intr['coeffs']                           
    
    depth_roi = depth_frame[range(int(point[1])-10, int(point[1])+10), range(int(point[0])-10, int(point[0])+10)]
    depth = np.average(depth_roi[np.nonzero(depth_roi)])
    x = (point[0] - ppx) / fx
    y = (point[1] - ppy) / fy
    r2 = x * x + y * y
    f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2 
    ux = x * f + 2*coeffs[2]*x*y + coeffs[3]*(r2 + 2*x*x)
    uy = y * f + 2*coeffs[3]*x*y + coeffs[2]*(r2 + 2*y*y)
    x = ux
    y = uy
    x_ = depth*x
    y_ = depth*y
    z_ = depth
    
    return [int(x_), int(y_), int(z_)]

#calculate the position by math model(wxm)
def coor_transform_through_geometry(center_points, depth_data):
    H = 345                 
    Dmin = 110             
    Dmax = 628      
    pi = math.pi      
    Beta =  50 * pi / 180
    height = 480
    width = 848
    D1 = 385
    fx = 624.3427734375
    fy = 624.3428344726562
    ux = 305.03887939453125
    vy = 244.86605834960938

    '''depth'''
    real_z	=  depth_data
    real_x	= (center_points[0] - 320 - ux) / fx * real_z
    real_y  = (center_points[1] - 240 - vy) / fy * real_z
    print ('real_y & real_x:', real_y, real_x)

    '''wxm'''
    Alpha = math.atan(Dmin/H)
    Zeta = math.atan(Dmax/H)-Alpha
    Zeta_stride = (height - center_points[1]) * Zeta / height

    YY = H * math.tan(Alpha + Zeta_stride)
    BB = (D1 + YY) * math.tan(Beta / 2)
    XX = -2 * BB * (center_points[0] - width / 2) / width
    print ('YY & XX:', YY, XX)

    '''mi'''
    Y1 = np.sqrt(depth_data ** 2 - H ** 2)
    B1 = (D1 + Y1) * math.tan(Beta / 2)
    X1 = -2 * B1 * (center_points[0] - width / 2) / width
    print ('Y1 & X1:', Y1, X1)

    print ('Y_real & X_real:', Y1 + 250, XX)

    return Y1 + 250, XX

def calculateCoM(dpt):   #计算质心
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < 0] = 0
    dc[dc > 10000] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    print ('cc:', cc)
    num = np.count_nonzero(dc)
    com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float)

    if num == 0:
        return np.array((0, 0, 0), np.float)
    else:
        return com/num

def pick_max_contour_inx(boxes):
    if len(boxes) == 0:
        print ('no contour box input')
        return []

    areas = []
    for i in range (len(boxes)):
        box0 = boxes[i][0]
        box1 = boxes[i][1]
        box2 = boxes[i][2]
        box3 = boxes[i][3]

        box_length = math.sqrt((box1[0] - box0[0]) ** 2 + (box1[1] - box0[1]) ** 2)
        box_width = math.sqrt((box3[0] - box0[0]) ** 2 + (box3[1] - box0[1]) ** 2)

        area = box_length * box_width
        areas.append(area)
    
    pick_inx = np.argmax(areas)
    
    return pick_inx

def contour_detect_test(testImg, bbox): #边缘检测
    img = cv2.imread(testImg)
    img_region = img[bbox[0][1] - 50 : bbox[0][3] + 50, bbox[0][0] - 50 : bbox[0][2] + 50]
    img_gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)

    thresh, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    
    rects, boxes, angles = [], [], []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        # if rect[1][0] > 30 and rect[1][1] > 30:
        rects.append(rect)
        
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        boxes.append(box)

        angle = rect[2]
        angles.append(angle)

    pick_inx = pick_max_contour_inx(boxes)
    picked_center = rects[pick_inx][0]
    angle_picked = angles[pick_inx]
    print (angle_picked)

    box_picked = boxes[pick_inx]
    print (box_picked)
    cv2.drawContours(img, [box_picked], 0, (0, 0, 255), 5)
    
    cv2.imshow('CONTOUR_DETECT', img_region)

if __name__ == '__main__':
    color_image, depth_image = get_aligned_imgs()
    time.sleep(0.5)
    
    testImg = '/home/toke/Desktop/image/camera_catch.png'  #跑模型+相机的照片
    #testImg = '/home/toke/Desktop/image/1.jpeg'  #跑模型+相机的照片
    #testImg = '/home/toke/Desktop/image-d-test/2.jpg'   #深度值聚类测试
    #testImg = '/home/toke/Desktop/image-test/2.jpg'   #边缘检测+角度获取测试
    bbox_pick, center_points, bbox1 = bbox_yolov4(testImg)                   # python3 run pyyolo

    depth_crop = depth_image[int(bbox_pick[1]):int(bbox_pick[3]), int(bbox_pick[0]):int(bbox_pick[2])] #bbox区域
    img_color = cv2.imread(testImg) 
    color_crop = img_color[int(bbox_pick[1]):int(bbox_pick[3]), int(bbox_pick[0]):int(bbox_pick[2])]

    # crop_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_crop, alpha=0.03), cv2.COLORMAP_JET)
    # cv2.imshow("cropped depth", crop_colormap)
    # cv2.waitKey(1000)
    #depth_crop1 = depth_crop * 255


    cv2.imwrite('/home/toke/Desktop/image/color_crop.png',color_crop) #保存二值图
    plt.imshow(depth_crop, cmap='gray')  #显示最后一个框的深度图（full image的局部）
    plt.savefig("/home/toke/Desktop/image/depth_crop.png")
    plt.show()

    #contour_detect('/home/toke/Desktop/image/color_crop.png')
    print('中心点深度值')
    print(depth_image[int(center_points[1])][int(center_points[0])])   #输出框中心的深度值

#-----------------------------------深度值对照组---------------------------------------------#
   
    mass_center = calculateCoM(depth_crop)  #计算质心
    depth_data = mass_center[2]
    print('质心深度值')
    print ('mass_center:', mass_center)   #输出质心坐标和对应的深度值

#------------------------------------------------------------------------------------------# 

    #image[y][x]   center_points[x,y]  注意在看看，image和中心坐标的x，y对应情况
    #y = depth_image[int((center_points[1]-10)):int((center_points[1]+10)), int((center_points[0]-10)):int((center_points[0]+10))]

#------------------------------------------------------------------------------------------# 
   
    y = depth_image[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]  #深度聚类（全框）
    y = y.reshape(-1,1)
    tmp = np.copy(y)
    y_clean = detect_outliers(tmp, 1.5) #过滤过的深度值
    y_clean = y_clean.reshape(-1,1)
    # y_clean = detect_outliers(y_clean, 0.1) #过滤过的深度值
    # y_clean = y_clean.reshape(-1,1)
    
    print(np.shape(y))
    print(np.shape(y_clean))
    #k_silhouette(y, 10)
    #k_silhouette(y_clean, 10)
    labels=['Depth_data']
    flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}
    #flierprops = {'marker':'o','color':'red'}
    plt.boxplot(y, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
    plt.show()
    # plt.boxplot(y_clean, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
    # plt.show()


    #depth_data_KMeans(y, 3)
    depth_data_KMeans(y_clean, 3)
    # print(y)
    # print(y_del)





    xx, yy = coor_transform_through_geometry(center_points, depth_data)  #坐标转换
    theta = 8.00  
    theta = 8.00 
    theta = 8.00 