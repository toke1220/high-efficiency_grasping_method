# coding:utf-8
''' this code is used to acquire object coordinate from camera frame through geometry calculation, and add contour detection. completed on 20, July.'''

from PIL import Image, ImageFont, ImageDraw
import pyrealsense2 as rs
from scipy import ndimage
import numpy as np
import glob as gb
import pyyolo
import math
import time
import os

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

import binascii
import struct
import serial 

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #848*480
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

def get_aligned_imgs():
    for i in range(50):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imwrite('/home/toke/Desktop/img/camera_catch.jpg', color_image)

    return color_image, depth_image

def bbox_yolov3(testImg):
    darknet_path = '/home/toke/projects/darknet'
    datacfg = '/home/toke/projects/pyyolo/darknet/cfg/coco.data'
    cfgfile = '/home/toke/projects/pyyolo/darknet/cfg/yolov3.cfg'
    weightfile = '/home/toke/projects/pyyolo/yolov3.weights'

    thresh = 0.45 # 0.24
    hier_thresh = 0.5  # 0.5
    pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)
    outputs = pyyolo.test(testImg, thresh, hier_thresh, 0)

    img = Image.open(testImg)
    fname = os.path.basename(testImg)
    draw = ImageDraw.Draw(img)

    bbox_out = []
    for output in outputs:
        bbox = (output['left']), (output['top']), (output['right']), (output['bottom'])
        bbox_out.append(bbox)

    bbox_pick = np.vstack(bbox_out)
    print ('object bounding box:', bbox_pick)

    center_points = []
    for i in range(len(bbox_pick)):
        lt_pos = (bbox_pick[i][0], bbox_pick[i][1])
        bm_pos = (bbox_pick[i][2], bbox_pick[i][3])
        draw.rectangle([lt_pos, bm_pos], outline=(255, 0, 0))
        
        center_point = (math.ceil((bbox_pick[i][2] + bbox_pick[i][0]) / 2), math.ceil((bbox_pick[i][3] + bbox_pick[i][1]) / 2))
        draw.ellipse((center_point[0], center_point[1], center_point[0] + 20, center_point[1] + 20), 'red')
        
    center_points = np.vstack(center_point)
    print ('center points:', center_points)
    
    img.show()
    path = '/home/toke/Desktop/img_box'
    fname = os.path.basename(testImg)
    img.save(os.path.join(path, fname))

    return bbox_pick, center_points

def bbox_yolov4(testImg):
    ### python3 run yolov4 model
    img = Image.open(testImg)
    fname = os.path.basename(testImg)
    draw = ImageDraw.Draw(img)
    image = cv2.imread(testImg)

    detector = pyyolo.YOLO("/home/toke/projects/PyYOLO/models/yolov4-custom.cfg",
                           "/home/toke/projects/PyYOLO/models/yolov4-custom_45000_2020_0629.weights",
                           "/home/toke/projects/PyYOLO/models/my_laji.data",
                           detection_threshold = 0.5,
                           hier_threshold = 0.5,
                           nms_threshold = 0.45)

    bboxes, center_points = [], []
    dets = detector.detect(image, rgb=False)
    for i, det in enumerate(dets):
        # print ('Detection:', {i}, {det})
        xmin, ymin, xmax, ymax = det.to_xyxy()
        bbox = [xmin, ymin, xmax, ymax]

        center_point = (math.ceil((xmin + xmax) / 2), math.ceil((ymin + ymax) / 2))
        draw.ellipse((center_point[0], center_point[1], center_point[0] + 20, center_point[1] + 20), 'red')

    bboxes = np.vstack(bbox)
    center_points = np.vstack(center_point)

    img.show()
    path = '/home/toke/Desktop/img'
    fname = os.path.basename(testImg)
    img.save(os.path.join(path, fname))

    return bboxes, center_points

def coor_transform_through_geometry(center_points, depth_data):
    H = 369                 
    Dmin = 167             
    Dmax = 832      
    pi = math.pi      
    Beta =  50 * pi / 180
    height = 480
    width = 640
    D1 = 362
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

    print ('Y_real & X_real:', Y1 + 180, XX)

    return Y1 + 180, XX

def calculateCoM(dpt):
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

def contour_detect(testImg, bbox):
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

def float_2_char_buff(des_buff, src_data):
    temp = struct.pack("<f", src_data)
    for idx in range(4):
        des_buff.append(temp[idx])

def check_sum8(src_buff):
    sumValue = 0xff
    
    for idx in range(2, len(src_buff)):
        sumValue += src_buff[idx]
    
    return sumValue % 256                           # check sum 8

def coordinate_data_pack(des_buff, xx, yy, theta):
    float_2_char_buff(des_buff, xx)
    float_2_char_buff(des_buff, yy)
    float_2_char_buff(des_buff, theta)
    # des_buff[2] = len(des_buff) - 2                   # fill the length byte, exclusive HEAD Frame: AA 55
    des_buff.append(check_sum8(des_buff))           # calculate the check sum value
    
    print ([hex(des_buff[i]) for i in range(len(des_buff))])

    return struct.pack("%dB" % (len(des_buff)), *des_buff)

if __name__ == '__main__':
    color_image, depth_image = get_aligned_imgs()
    time.sleep(2)
    
    testImg = '/home/toke/Desktop/img/camera_catch.jpg'
    bbox_pick, center_points = bbox_yolov3(testImg)                     # python2 run pyyolo
    #bbox_pick, center_points = bbox_yolov4(testImg)                   # python3 run pyyolo

    # bbox_pick = np.array([[220, 120, 621, 245]])
    depth_crop = depth_image[int(bbox_pick[0][1]):int(bbox_pick[0][3]), int(bbox_pick[0][0]):int(bbox_pick[0][2])]
    crop_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_crop, alpha=0.03), cv2.COLORMAP_JET)
    # cv2.imshow("cropped depth", crop_colormap)
    # cv2.waitKey(1000)

    mass_center = calculateCoM(depth_crop)
    depth_data = mass_center[2]
    print ('mass_center:', mass_center)

    xx, yy = coor_transform_through_geometry(center_points, depth_data) 
    theta = 8.00  

    send_buff = []
    send_buff.append(0xAA)
    send_buff.append(0x55)
    send_buff.append(0x02)

    data = coordinate_data_pack(send_buff, xx, yy, theta)                # pack the coordinate data
    
    try:
        portx = "/dev/ttyUSB0"
        bps = 115200
        timex = 5
        
        ser=serial.Serial(portx, bps, timeout=timex)
        result=ser.write(data)
        print("transmit length:", result)

        ser.close()
    
    except Exception as e:
        print("---serial transmit error---：", e)
