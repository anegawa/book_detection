import cv2, math
from math import *
import numpy as np
import sys
from decimal import *
sys.path.append("../")
from libs.configs import cfgs
IMG_LOW = 1100
black = (0,0,0)
red = (0, 0, 255)

def convert_rect_origin(rect):
    if rect[4] == 90 or rect[4] == -90:
        new_rect = [rect[0],rect[1],rect[3],rect[2], 0]
    elif rect[4] > 0:
        new_rect = [rect[0], rect[1], rect[3], rect[2], -90 + rect[4]]
        # new_rect = (rect[0], (rect[1][1], rect[1][0]), -90 + rect[2])
    elif rect[4] < 0:
        new_rect = [rect[0],rect[1], rect[3], rect[2], 90 + rect[4]]
        # new_rect = (rect[0], (rect[1][1], rect[1][0]), 90 + rect[2])
    else: #rect[2] == 0
        new_rect = [rect[0],rect[1], rect[3], rect[2], 0]
        # new_rect = (rect[0], (rect[1][1], rect[1][0]), 0)
    return new_rect

#boxの向きを写真の下までの直線の点を得る
def check_rect_line(img, rect):
    # print(img.shape)
    bb = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    # print(rect)
    # print(img.shape)
    if bb[2] > 0:
        flag = 1
    else:
        flag = -1
    x = bb[0][0]
    yoko, tate = hanbun(bb)
    h = img.shape[0]-rect[1]
    if bb[2] == 0 or abs(bb[2]) == 90:
        a = x - yoko
        b = x + yoko
    else:
        a = x + flag * h*cos(radians(abs(bb[2])))/sin(radians(abs(bb[2])))
        b = x + flag * -1 * yoko

    if a < b:
        xmin = a
        xmax = b
    else:
        xmin = b
        xmax = a
    # img = cv2.line(img,(bb[0][0],bb[0][1]),(int(a),924),(0,255,0),5)
    return img, a

#この関数に投げたら、適したboxに変更してくれる
def check_rect(img, rect):
    rect2 = convert_rect_origin(rect)
    img, a  = check_rect_line(img, rect)
    img, a2 = check_rect_line(img, rect2)
    diff  = abs(a  - rect[0])
    diff2 = abs(a2 - rect[0])
    if diff > diff2:
        return img, rect2
    else:
        return img, rect

def calc_x_range(img, rect):
    # print(img.shape)
    x_min_maxs = []
    for i in range(cfgs.STRIDE_NUM):
        img, rect = check_rect(img, rect)
        bb = ((rect[0] + (i-2) * cfgs.STRIDE , rect[1]), (rect[2], rect[3]), rect[4])
        # print(rect)
        # print(img.shape)
        if bb[2] > 0:
            flag = 1
        else:
            flag = -1
        x = bb[0][0]
        yoko, tate = hanbun(bb)
        h = IMG_LOW-rect[1]
        thre = 950
        if h > thre:
            h = thre
        if bb[2] == 0 or abs(bb[2]) == 90:
            a = x - yoko
            b = x + yoko
        else:
            # a = x + flag * (img.shape[0]-rect[1])*cos(radians(abs(bb[2])))/sin(radians(abs(bb[2])))
            a = x + flag * h *cos(radians(abs(bb[2])))/sin(radians(abs(bb[2])))
            b = x + flag * -1 * yoko

        if a < b:
            xmin = a
            xmax = b
        else:
            xmin = b
            xmax = a
        x_min_maxs.append([xmin, xmax])
    img = cv2.line(img,(bb[0][0],bb[0][1]),(int(a),950),(0,255,0),5)
    return img, x_min_maxs

def hanbun(rect):
    if rect[2] == 0:
        yoko_hanbun = 0.5*rect[1][0]
        tate_hanbun = 0.5*rect[1][1]
    elif rect[2] == 90 or rect[2] == -90:
        yoko_hanbun = 0.5*rect[1][1]
        tate_hanbun = 0.5*rect[1][0]
    else:
        yoko_hanbun = 0.5*(rect[1][0]*cos(radians(abs(rect[2]))) + rect[1][1]*sin(radians(abs(rect[2]))))
        tate_hanbun = 0.5*(rect[1][0]*sin(radians(abs(rect[2]))) + rect[1][1]*cos(radians(abs(rect[2]))))
	# elif rect[2] < 0:
	# 	yoko_hanbun = 0.5*(rect[1][0]*cos(radians(abs(rect[2]))) + rect[1][1]*sin(radians(abs(rect[2]))))
	# 	tate_hanbun = 0.5*(rect[1][0]*sin(radians(abs(rect[2]))) + rect[1][1]*cos(radians(abs(rect[2]))))
	# else:
	# 	yoko_hanbun = 0.5*(rect[1][1]*sin(radians(abs(rect[2]))) + rect[1][0]*sin(radians(abs(rect[2]))))
	# 	tate_hanbun = 0.5*(rect[1][1]*cos(radians(abs(rect[2]))) + rect[1][0]*cos(radians(abs(rect[2]))))
    return [yoko_hanbun, tate_hanbun]

def convert_rect(rect):
	if rect[2] == 90 or rect[2] == -90:
		new_rect = (rect[0], (rect[1][1], rect[1][0]), 0)
	elif rect[2] > 0:
		new_rect = (rect[0], (rect[1][1], rect[1][0]), -90 + rect[2])
	elif rect[2] < 0:
		new_rect = (rect[0], (rect[1][1], rect[1][0]), 90 + rect[2])
	else: #rect[2] == 0
		new_rect = (rect[0], (rect[1][1], rect[1][0]), 0)
	return new_rect

def calc_book_range(img, rect):
    return_boxes = []
    for i in range(cfgs.STRIDE_NUM):
        print(rect)
        img, rect = check_rect(img, rect)
        print(rect)
        bb = ((rect[0] + (i-2) * cfgs.STRIDE , rect[1]), (rect[2], rect[3]), rect[4])
        if bb[2] > 0:
            flag = 1
        else:
            flag = -1
        x,y = bb[0][0], bb[0][1]
        yoko, tate = hanbun(bb)
        print("tate : " + str(tate))
        print("yoko : " + str(yoko))
        h2 = IMG_LOW-rect[1]


        thre = img.shape[0]
        print("tres : " + str(thre))

        if h2 > thre:
            h2 = thre
        if bb[2] == 0 or abs(bb[2]) == 90:
            if bb[2] == 0:
                bb = (bb[0], (bb[1][1], bb[1][0]), 90)
            h1 = tate
            line_h = h1 + h2
            line_w = 1000000
            cnt_x = bb[0][0]
            cnt_y = bb[0][1] + line_h/2-h1
            line_norm = line_h

        else:
            h1 = yoko * tan(radians(abs(bb[2])))
            line_h = h1 + h2
            line_w = line_h / tan(radians(abs(bb[2])))
            line_norm = line_h / sin(radians(abs(bb[2])))
            cnt_x = x + flag * (-yoko + line_w /2)
            cnt_y = y - h1 + line_h /2


        print("cnt_x : " + str(cnt_x))
        print("cnt_y : " + str(cnt_y))
        print("line_h : " + str(line_h))
        print("line_w : " + str(line_w))
        print("line_norm : " + str(line_norm))
        img = cv2.line(img, (0, int(cnt_y + line_h/2)), (800, int(cnt_y + line_h/2)), (255,0,0), 5)
        img = cv2.circle(img,(int(cnt_x), int(cnt_y)), 6, (0,255,0), -1)

        box1 = ((cnt_x, cnt_y), (line_norm, bb[1][1]), bb[2])
        # box1 = [cnt_x, cnt_y, bb[1][1], line_norm,bb[2]]
        # img, box1 = check_rect(img, box1)
        # box1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        box = cv2.boxPoints(box1)
        box = np.int0(box)
        img = cv2.drawContours(img,[box],-1,black,2)
        return_boxes.append(box1)
    # img = cv2.line(img,(bb[0][0],bb[0][1]),(int(a),950),(0,255,0),5)
    return img, return_boxes


# rect = [500, 500, 50, 100, 80]
# rect1= [503, 800, 100, 50, 10]
# rect2 = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
# rect1 = ((rect1[0], rect1[1]), (rect1[2], rect1[3]), rect1[4])
#
# img = cv2.imread("hon.jpg")
# img = cv2.circle(img,(int(rect1[0][0]), int(rect1[0][1])), 3, (0,255,0), -1)
# img,x = calc_x_range(img, rect)
# box = cv2.boxPoints(rect2)
# box = np.int0(box)
# img = cv2.drawContours(img,[box],-1,red,2)
# box = cv2.boxPoints(rect1)
# box = np.int0(box)
# img = cv2.drawContours(img,[box],-1,red,2)
# img, boxes = calc_book_range(img, rect)
# area_bb2 = rect1[1][0] * rect1[1][1]
#
# for i in range(len(boxes)):
#     int_pts = cv2.rotatedRectangleIntersection(boxes[i], rect1)[1]
#     inter = 0.0
#     if int_pts is not None:
#         #convexhull は　凸法を計算
#         order_pts = cv2.convexHull(int_pts, returnPoints=True)
#         #order_ptsの面積を計算
#         int_area = cv2.contourArea(order_pts)
#         inter = int_area * 1.0 / area_bb2
#         print(inter)
# # cv2.imshow("te3", img)
# cv2.imwrite("res.jpg", img)
# cv2.waitKey(0)

# rect = np.array([500, 500, 50, 100, 80])
# rect1= np.array([503, 800, 100, 50, 10])
rect = np.array([1,1])
rect1= np.array([1,1])
rects = np.array([rect, rect1])
print(np.var(rects))
