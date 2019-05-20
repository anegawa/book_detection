# -*- coding: utf-8 -*-
import numpy as np

import os, sys

import math
from math import *
from PIL import Image, ImageDraw, ImageFont
import cv2
from sympy import *
from os import environ
import tensorflow as tf

from libs.configs import cfgs
from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from libs.box_utils.nms_rotate import *
from libs.box_utils.iou import *


NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
]
FONT = ImageFont.load_default()
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0,0,0)
white = (255,255,255)
WIDE = 0
EXTEND_RECD = False
RECT_SIZE = 20


METHOD_NAME = ['Correlation', 'Chi-square', 'Intersection', 'Bhattacharyya distance']
METHOD_INDEX = 3
METHOD = int(environ.get('METHOD', METHOD_INDEX))
# THRESHOLD = 4000
if METHOD_INDEX == 3:
    THRESHOLD = 0.25
elif METHOD_INDEX == 2:
    THRESHOLD = 1000
else:
    sys.exit(1)
IMG_LOW = 935
GAUSS_FLAG = True
print("WIDE : " + str(WIDE))
print("THRESHOLD : " + str(THRESHOLD))
print("IMG_LOW : " + str(IMG_LOW))
print("GAUSS_FLAG : " + str(GAUSS_FLAG))
print("METHOD : " + str(METHOD_NAME[METHOD_INDEX]))

def draw_a_rectangel_in_img(draw_obj, box, color, width):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    top_left, top_right = (x1, y1), (x2, y1)
    bottom_left, bottom_right = (x1, y2), (x2, y2)
    draw_obj.line(xy=[top_left, top_right],fill=color,width=width)
    draw_obj.line(xy=[top_left, bottom_left],fill=color,width=width)
    draw_obj.line(xy=[bottom_left, bottom_right],fill=color,width=width)
    draw_obj.line(xy=[top_right, bottom_right],fill=color,width=width)


def only_draw_scores(draw_obj, box, score, color):

    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x+60, y+10],
                       fill=color)
    draw_obj.text(xy=(x, y),
                  text="obj:" +str(round(score, 2)),
                  fill='black',
                  font=FONT)


def draw_label_with_scores(draw_obj, box, label, score, color):
    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                       fill=color)

    txt = LABEl_NAME_MAP[label] + ':' + str(round(score, 2))
    draw_obj.text(xy=(x, y),
                  text=txt,
                  fill='black',
                  font=FONT)


def draw_boxes_with_label_and_scores(img_array, boxes, labels, scores):
    img_array = img_array + np.array(cfgs.PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):

        if a_label != NOT_DRAW_BOXES:
            num_of_objs += 1
            draw_a_rectangel_in_img(draw_obj, box, color=STANDARD_COLORS[a_label], width=3)
            if a_label == ONLY_DRAW_BOXES:  # -1
                continue
            elif a_label == ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                 only_draw_scores(draw_obj, box, a_score, color='White')
                 continue
            else:
                draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

    return np.array(out_img_obj)


def draw_box_cv(img, boxes, labels, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        # print("=========")
        # print(box)
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

            category = LABEl_NAME_MAP[label]

            # if scores is not None:
            #     cv2.rectangle(img,
            #                   pt1=(xmin, ymin),
            #                   pt2=(xmin+150, ymin+15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category+": "+str(scores[i]),
            #                 org=(xmin, ymin+10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
            # else:
            #     cv2.rectangle(img,
            #                   pt1=(xmin, ymin),
            #                   pt2=(xmin + 40, ymin + 15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category,
            #                 org=(xmin, ymin + 10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img


def draw_rotate_box_cv(img, boxes, labels, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    vct = []

    num_of_object = 0
    for i, box in enumerate(boxes):
        # print("------------")
        # print(box)
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        cv2.putText(img, text=str(box[0]) + "," + str(box[1]), org=(box[0], box[1]), fontFace=3, fontScale=0.5, color=(255,255,255))
        label = labels[i]
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            rect = ((x_c, y_c), (w, h), theta)
            # rect = ((x_c, y_c), (w, h), 45)

            #類似度
            # if theta > 45:
            #     theta = 90 - theta
            # if theta < -45:
            #     theta = -90 - theta
            # b = theta*x_c - y_c
            # vct.add([theta, b])

            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(img, [rect], -1, color, 2)

            category = LABEl_NAME_MAP[label]

            # if scores is not None:
            #     cv2.rectangle(img,
            #                   pt1=(x_c, y_c),
            #                   pt2=(x_c + 120, y_c + 15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category+": "+str(scores[i]),
            #                 org=(x_c, y_c+10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
            # else:
            #     cv2.rectangle(img,
            #                   pt1=(x_c, y_c),
            #                   pt2=(x_c + 40, y_c + 15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category,
            #                 org=(x_c, y_c + 10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img


def cos_sim(v1, v2):
    # print(v1)
    # print(v2)
    # return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    v = np.array(v1) - np.array(v2)
    s = 0
    for i in range(len(v)):
        s = s + v[i]**2
    return sqrt(s)
    # return v1 - v2

#矩形を別の見方で見るよ
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

#矩形を別の見方で見るよ オリジナルのボックス
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

#縦横半分を返すよ
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


def norm(x, axis=None):
    tmp = add.reduce((x.conj() * x).real, axis=axis)
    if type(tmp) == float:
        return math.sqrt(tmp)
    else:
        return np.sqrt(tmp.astype(np.float64))

#rect is red
#rect_other is green
#rect_result is black
#new version
def connect_rect_new(img, rects, verbose = True): #できれば可変長の引数のほうがいいか
	#rect = ((300, 500), (100, 50), -50) #left
    if verbose:
        print("=============== connect ===============")
    rect_num = len(rects)
    max_u = [9999, 1000]
    max_d = [0, 1000]
    max_r = [0, 1000]
    max_l = [9999, 1000]
    if verbose:
        print("rect_num : " + str(rect_num))
    plus_angle = 0
    minus_angle = 0
    zero_angle = 0
    for i,rect in enumerate(rects):
        if verbose:
            print(rect)
        rect_draw = cv2.boxPoints(rect)
        rect_draw = np.int0(rect_draw)
        img = cv2.drawContours(img, [rect_draw], -1, blue, 2)
        if verbose:
            print("rect" + str(i) + " : " + str(rect))
        # if rect[2] > 0:
        if rect[2] > 0:
            plus_angle += 1
        elif rect[2] < 0:
            minus_angle += 1
        else:
            zero_angle += 1
        if rect[2] < 0:
            yoko_hanbun = 0.5*(rect[1][0]*cos(radians(abs(rect[2]))) + rect[1][1]*sin(radians(abs(rect[2]))))
            tate_hanbun = 0.5*(rect[1][0]*sin(radians(abs(rect[2]))) + rect[1][1]*cos(radians(abs(rect[2]))))
            r = rect[0][0] + yoko_hanbun
            l = rect[0][0] - yoko_hanbun
            u = rect[0][1] - tate_hanbun
            d = rect[0][1] + tate_hanbun
        else:
            yoko_hanbun = 0.5*(rect[1][1]*sin(radians(abs(rect[2]))) + rect[1][0]*sin(radians(abs(rect[2]))))
            tate_hanbun = 0.5*(rect[1][1]*cos(radians(abs(rect[2]))) + rect[1][0]*cos(radians(abs(rect[2]))))
            r = rect[0][0]+yoko_hanbun
            l = rect[0][0]-yoko_hanbun
            u = rect[0][1]-tate_hanbun
            d = rect[0][1]+tate_hanbun
        if max_r[0] <= r:
            max_r = [r, i]
        if max_l[0] >= l:
            max_l = [l, i]
        if max_u[0] >= u:
            max_u = [u, i]
        if max_d[0] <= d:
            max_d = [d, i]
		# else:
		# 	r = rect[0][0]+0.5*(rect[1][0]*math.sin(math.radians(abs(rect[2]))) + rect[1][1]*math.cos(math.radians(abs(rect[2]))))
		# 	l = rect[0][0]-0.5*(rect[1][1]*math.cos(math.radians(abs(rect[2]))) + rect[1][0]*math.sin(math.radians(abs(rect[2]))))
		# 	u = rect[0][1]+0.5*(rect[1][1]*math.sin(math.radians(abs(rect[2]))) + rect[1][0]*math.cos(math.radians(abs(rect[2]))))
		# 	d = rect[0][1]-0.5*(rect[1][1]*math.sin(math.radians(abs(rect[2]))) + rect[1][0]*math.cos(math.radians(abs(rect[2]))))
		# 	if max_r[0] <= (rect[0][0]+0.5*rect[1][0]*math.sin(math.radians(abs(rect[2])))):
		# 		max_r = [rect[0][0]+0.5*rect[1][0]*math.sin(math.radians(abs(rect[2]))), i]
    rect_u = rects[max_u[1]]
    rect_d = rects[max_d[1]]
    rect_l = rects[max_l[1]]
    rect_r = rects[max_r[1]]

	#使わないかも・・・
    if plus_angle > minus_angle:
        flag = True
    else:
        flag = False
    if rect_u[2] > 0:
        u_p_x = rect_u[0][0] - hanbun(rect_u)[0] + rect_u[1][1] * sin(radians(abs(rect_u[2])))
    else:
        u_p_x = rect_u[0][0] - hanbun(rect_u)[0] + rect_u[1][0] * cos(radians(abs(rect_u[2])))
    u_point = [u_p_x, max_u[0]]
    img = cv2.circle(img,(int(u_point[0]), int(u_point[1])), 6, green, -1)

    if rect_l[2] > 0:
        l_p_y = rect_l[0][1] - hanbun(rect_l)[1] + rect_l[1][1]*cos(radians(abs(rect_l[2])))
    else:
        l_p_y = rect_l[0][1] - hanbun(rect_l)[1] + rect_l[1][0]*sin(radians(abs(rect_l[2])))
    l_point = [max_l[0], l_p_y]
    img = cv2.circle(img,(int(l_point[0]), int(l_point[1])), 6, white, -1)

    if rect_d[2] > 0:
        d_p_x = rect_d[0][0] - hanbun(rect_d)[0] + rect_d[1][0] * sin(radians(abs(rect_d[2])))
    else:
        d_p_x = rect_d[0][0] - hanbun(rect_d)[0] + rect_d[1][1] * sin(radians(abs(rect_d[2])))
    d_point = [d_p_x, max_d[0]]
    img = cv2.circle(img,(int(d_point[0]), int(d_point[1])), 6, black, -1)

    if rect_r[2] > 0:
        r_p_y = rect_r[0][1] - hanbun(rect_r)[1] + rect_r[1][0]*cos(radians(abs(rect_r[2])))
    else:
        r_p_y = rect_r[0][1] - hanbun(rect_r)[1] + rect_r[1][1]*cos(radians(abs(rect_r[2])))
    r_point = [max_r[0], r_p_y]
    img = cv2.circle(img,(int(r_point[0]), int(r_point[1])), 6, red, -1)
    x = (r_point[0] + l_point[0])/2
    y = (u_point[1] + d_point[1])/2

	# print(np.array(u_point) - np.array(l_point))
	# print(np.array(u_point) - np.array(r_point))
	# print(np.array(d_point) - np.array(r_point))
	# print(np.array(d_point) - np.array(l_point))
    ln_ul = Line(Point(l_point[0], l_point[1]), Point(u_point[0], u_point[1]))
    ln_ur = Line(Point(r_point[0], r_point[1]), Point(u_point[0], u_point[1]))
    ln_dr = Line(Point(r_point[0], r_point[1]), Point(d_point[0], d_point[1]))
    ln_dl = Line(Point(l_point[0], l_point[1]), Point(d_point[0], d_point[1]))
    # norm_ul = norm(np.array(u_point) - np.array(l_point))
	# norm_ur = norm(np.array(u_point) - np.array(r_point))
	# norm_dr = norm(np.array(d_point) - np.array(r_point))
	# norm_dl = norm(np.array(d_point) - np.array(l_point))
    norm_ul = np.linalg.norm(np.array(u_point).astype(np.float32) - np.array(l_point).astype(np.float32))
    norm_ur = np.linalg.norm(np.array(u_point).astype(np.float32) - np.array(r_point).astype(np.float32))
    norm_dr = np.linalg.norm(np.array(d_point).astype(np.float32) - np.array(r_point).astype(np.float32))
    norm_dl = np.linalg.norm(np.array(d_point).astype(np.float32) - np.array(l_point).astype(np.float32))

    if norm_ur > norm_dl:
        w = norm_ur
    else:
        w = norm_dl
    if norm_ul > norm_dr:
        h = norm_ul
    else:
        h = norm_dr

    vct_ul = (np.array(u_point) - np.array(l_point)).astype(np.float32)
    vct_axis_aligned = np.array([100, 0]).astype(np.float32)
    i = np.inner(vct_ul, vct_axis_aligned)
    n = np.linalg.norm(vct_ul) * np.linalg.norm(vct_axis_aligned)
    c = i / n
    theta = -np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
    if theta > 90:
        theta = -180 + theta
    elif theta < -90:
        theta = 180 + theta

    # h = r_point[0] - l_point[0]
    # w = img.shape[1]
    # theta = 0

    rect_result = ((x, y), (h, w), theta) #up
    rect_result = cv2.boxPoints(rect_result)
    rect_result = np.int0(rect_result)
    img = cv2.drawContours(img, [rect_result], -1, red, 2)
    if verbose:
        print("------new rect------")
        print("x     : " + str(x))
        print("y     : " + str(y))
        print("w     : " + str(w))
        print("h     : " + str(h))
        print("theta : " + str(theta))
        print("--------------------")
        print("=======================================")

    cv2.imwrite("a.png", img)
    return rect_result


def make_vct(boxes, labels):
    vct = []
    for i, box in enumerate(boxes):
        # print("------------")
        # print(box)
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        label = labels[i]
        if label != 0:
            # num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (255, 0, 0)
            # print("-----")
            # print(theta)
            # print(w, h)
            ha = 500
            if theta > 45 and w < h:
                theta = 90 - theta
                buf = h
                h = w
                w = buf
                h = ha
            elif theta < -45 and w < h:
                theta = -90 - theta
                buf = h
                h = w
                w = buf
                h = ha
            elif theta < 45 and w > h and theta > -45:
                if theta > 0:
                    theta = -180 + theta
                elif theta < 0:
                    theta = 180 + theta
                buf = h
                h = w
                w = buf
                h = ha
            else:
                if w > h:
                    color = (0,0,0)
                    # print(x_c ,y_c, w, h, theta)
                    buf = h
                    h = w
                    w = buf
                    h = ha
                    if theta > 0:
                        theta = -90 + theta
                    else:
                        theta = 90 - theta
                    # w = ha
                else:
                    color = (255,255,255)
                    h = ha
        box = [x_c, y_c, w, h, theta]
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        label = labels[i]
        if label != 0:
            rect = ((x_c, y_c), (w, h), theta)
            # rect = ((x_c, y_c), (w, h), 45)

            #類似度
            # if theta > 45:
            #     theta = 90 - theta
            # if theta < -45:
            #     theta = -90 - theta
            # print(theta)

            if theta > 0:
                flag = -1
            else:
                flag = 1

            if theta == 0:
                a = 9999
                b = x_c
                param = [1, 0, x_c]
            else:
                sin_v = math.sin(radians(abs(theta)))
                cos_v = math.cos(radians(abs(theta)))
                a = flag * sin_v / cos_v
                b = y_c - a*x_c
                x_const = (900 - b) / a
                param = [sin_v, cos_v, x_const]
            vct.append([param, i])
    return boxes, vct

#pointは左上から時計回りに1~4
def calc_4points(img, rect):
    # print(rect[2])
    yoko, tate = hanbun(rect)
    l = rect[0][0] - yoko
    r = rect[0][0] + yoko
    u = rect[0][1] - tate
    d = rect[0][1] + tate
    if rect[2] == 0:
        point1 = [l, u]
        point2 = [r, u]
        point3 = [r, d]
        point4 = [l, d]
    elif rect[2] == 90 or rect[2] == -90:
        point1 = [l, u]
        point2 = [r, u]
        point3 = [r, d]
        point4 = [l, d]
    elif rect[2] > 0:
        point1 = [l, u + rect[1][1] * cos(radians(abs(rect[2])))]
        point2 = [l + rect[1][1] * sin(radians(abs(rect[2]))), u]
        point3 = [r, d - rect[1][1] * cos(radians(abs(rect[2])))]
        point4 = [r - rect[1][1] * sin(radians(abs(rect[2]))), d]
    elif rect[2] < 0:
        point1 = [r - rect[1][1] * sin(radians(abs(rect[2]))), u]
        point2 = [r, u + rect[1][1] * cos(radians(abs(rect[2])))]
        point3 = [l + rect[1][1] * sin(radians(abs(rect[2]))), d]
        point4 = [l, d - rect[1][1] * cos(radians(abs(rect[2])))]
    else:
        print("-------エラーやないかい-------")
        print("theta : " + str(rect[2]))
        sys.exit()
    # points = [np.array(point1), np.array(point2), np.array(point3), np.array(point4)]
    point1 = [int(point1[0]), int(point1[1])]
    point2 = [int(point2[0]), int(point2[1])]
    point3 = [int(point3[0]), int(point3[1])]
    point4 = [int(point4[0]), int(point4[1])]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # img = cv2.drawContours(img,[box],-1,green,2)
    # img = cv2.circle(img,(point1[0], point1[1]), 6, green, -1)
    # img = cv2.circle(img,(point2[0], point2[1]), 6, red, -1)
    # img = cv2.circle(img,(point3[0], point3[1]), 6, blue, -1)
    # img = cv2.circle(img,(point4[0], point4[1]), 6, white, -1)
    points = [point1, point2, point3, point4]
    for i in range(len(points)):
        # points[i] = np.int32(np.points[i])
        points[i] = np.array(points[i])
    points = [np.array([point1]), np.array([point2]), np.array([point3]), np.array([point4])]
    return np.array(points)

#斜めの長方形の範囲をトリミング
def trim_rotateRect(img, rect):
    #回転変換行列の計算
    # print("==========")
    # print("img  : " + str(img.shape))
    # print("rect : " + str(rect))
    rotation_matrix = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
    size = tuple(np.array([img.shape[1], img.shape[0]]))
    img_rot = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_LINEAR)
    img_rot = np.float32(img_rot)
    img_rot_trim = cv2.getRectSubPix(img_rot, tuple(rect[1]), tuple(rect[0]))
    return img_rot_trim

#斜めの長方形のヒストグラムを計算したい
def rect_color_hist(img, rect):
    # #ぼかすお
    # img = cv2.bilateralFilter(img,9,75,75)
    #斜めの長方形の範囲をトリミング
    rect = ((rect[0][0], rect[0][1]), (rect[1][0]+ WIDE, rect[1][1]+WIDE), rect[2])
    img_rot_trim = trim_rotateRect(img, rect)
    #切り出した範囲のヒストグラムを計算する
    hist_b = cv2.calcHist([img_rot_trim],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([img_rot_trim],[1],None,[256],[0,256])
    hist_r = cv2.calcHist([img_rot_trim],[2],None,[256],[0,256])
    normal = np.array(img_rot_trim).shape[0] * np.array(img_rot_trim).shape[1] / 30000
    hist_b = np.array(hist_b) / normal
    hist_g = np.array(hist_g) / normal
    hist_r = np.array(hist_r) / normal
    # plt.xlim(0, 255)
    # plt.plot(hist_r, "-r", label="Red")
    # plt.plot(hist_g, "-g", label="Green")
    # plt.plot(hist_b, "-b", label="Blue")
    # plt.xlabel("Pixel value", fontsize=20)
    # plt.ylabel("Number of pixels", fontsize=20)
    # plt.legend()
    # plt.grid()
    # plt.show()
    return hist_b, hist_g, hist_r

#return xmin, xmax
#boxのグループ化をするためのx軸の範囲を計算する
def calc_x_range(img, rect):
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
    h = IMG_LOW-rect[1]
    thre = 500
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
    # img = cv2.line(img,(bb[0][0],bb[0][1]),(int(a),924),(0,255,0),5)
    return img, xmin, xmax

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

#input  : 基準のbox, 走査対象のboxes (imgはヒストグラム計算用)
#output : 基準のboxと同じグループに属するboxes, その他のboxes
def box_grouping(img, a_box, boxes):
    group_boxes = []
    other_boxes = []
    #基準ボックスの準備(ヒストグラムとｘの範囲)
    bb1 = ((a_box[0], a_box[1]), (a_box[2], a_box[3]), a_box[4])
    bb1_hist_b, bb1_hist_g, bb1_hist_r = rect_color_hist(img, bb1)


    
    img, bb1_min, bb1_max = calc_x_range(img, a_box)
    criterion_vct = np.array([0,1])
    bb1_vct = np.array([bb1[0][0], bb1[0][1]])
    # print("---------------------------")
    # print("bb1 : " + str(bb1))

    #boxes内のボックスひとつひとつと比較
    for box in boxes:
        bb2 = ((box[0], box[1]), (box[2], box[3]), box[4])
        bb2_hist_b, bb2_hist_g, bb2_hist_r = rect_color_hist(img, bb2)
        bb2_vct = np.array([bb2[0][0], bb2[0][1]])
        # print("bb2 : " + str(bb2))

        yoko, tate = hanbun(bb2)
        bb2_min = bb2[0][0] - yoko
        bb2_max = bb2[0][0] + yoko

        if bb1_min <= bb2_min and bb1_max >= bb2_max:
            diff = 1
        else:
            diff = -1
        # print(bb1_hist_b.shape)
        if METHOD_INDEX == 3:
            ret_b = 1 - cv2.compareHist(bb1_hist_b, bb2_hist_b, METHOD)
            ret_g = 1 - cv2.compareHist(bb1_hist_g, bb2_hist_g, METHOD)
            ret_r = 1 - cv2.compareHist(bb1_hist_r, bb2_hist_r, METHOD)
        elif METHOD_INDEX == 2:
            ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, METHOD)
            ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, METHOD)
            ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, METHOD)
        # ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, 0)
        # ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, 0)
        # ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, 0)
        # print(ret_b, ret_g, ret_r)
        diff_vct = bb2_vct - bb1_vct

        if diff_vct[1] <= 0:
            deg = "under 0"
            deg_flag = False
        else:
            norm_diff_vct= np.linalg.norm(diff_vct)
            norm_criterion_vct= np.linalg.norm(criterion_vct)
            naiseki = np.dot(diff_vct, criterion_vct)
            coscos = naiseki/(norm_diff_vct*norm_criterion_vct)
            rad = acos(coscos)
            deg = math.degrees(rad)
            if deg < 45:
                deg_flag = True
            else:
                deg_flag = False



        # print("-----------")
        # print(bb1)
        # print(bb2)
        # print(bb1_max)
        # print(bb1_min)
        # print(bb2_max)
        # print(bb2_min)
        # print(diff)
        # print(deg, deg_flag)
        # print(ret_b)
        # print(ret_g)
        # print(ret_r)

        if diff > 0 and ret_b > THRESHOLD and ret_g > THRESHOLD and ret_r > THRESHOLD and deg_flag:
            img = cv2.line(img,(bb1[0][0],bb1[0][1]),(bb2[0][0], bb2[0][1]),(0,0,255),5)
            group_boxes.append(box)
        else:
            other_boxes.append(box)

    return img, group_boxes, other_boxes

#ボックスのグループ化を行う
#注意:vctは使ってないので，使いたいときはちゃんとmake_vctを編集してね
def box_group(img, boxes, vct):
    groups = []
    others = boxes
    while others != []:
        stash = others[0]
        img, group, others = box_grouping(img, others[0], others[1:])
        flag = True
        i = 0
        num = len(group)
        while i < num:
            img, g, others = box_grouping(img, group[i], others)
            count = 0
            if g != []:
                group.extend(g)
                num += len(g)
            i += 1
        group.append(stash)
        if group != []:
            groups.append(group)
    return img, groups

#input  : 基準のbox, 走査対象のboxes (imgはヒストグラム計算用)
#output : 基準のboxと同じグループに属するboxes, その他のboxes
def box_grouping_last(img, a_box, boxes):
    group_boxes = []
    other_boxes = []
    #基準ボックスの準備(ヒストグラムとｘの範囲)
    bb1 = ((a_box[0], a_box[1]), (a_box[2], a_box[3]), a_box[4])
    bb1_hist_b, bb1_hist_g, bb1_hist_r = rect_color_hist(img, bb1)
    # img, bb1_min, bb1_max = calc_x_range(img, a_box)
    yoko, tate = hanbun(bb1)
    bb1_min = bb1[0][0] - yoko
    bb1_max = bb1[0][0] + yoko

    #boxes内のボックスひとつひとつと比較
    for box in boxes:
        bb2 = ((box[0], box[1]), (box[2], box[3]), box[4])
        bb2_hist_b, bb2_hist_g, bb2_hist_r = rect_color_hist(img, bb2)

        yoko, tate = hanbun(bb2)
        bb2_min = bb2[0][0] - yoko
        bb2_max = bb2[0][0] + yoko

        # diff = -1
        # if bb1_min > bb2_min:
        #     if bb1_min >= bb2_max:
        #         diff = 1
        # else:
        #     if bb2_min >= bb1_max:
        #         diff = 1
        if bb1_min <= bb2_min and bb1_max >= bb2_max:
            diff = 1
        else:
            diff = -1

        ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, METHOD)
        ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, METHOD)
        ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, METHOD)
        # ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, 0)
        # ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, 0)
        # ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, 0)
        # print("-----------")
        # print(bb1)
        # print(bb2)
        # print(bb1_max)
        # print(bb1_min)
        # print(bb2_max)
        # print(bb2_min)
        # print(diff)
        # print(ret_b)
        # print(ret_g)
        # print(ret_r)

        if diff > 0:
        # if diff > 0 and ret_b > THRESHOLD and ret_g > THRESHOLD and ret_r > THRESHOLD:
            img = cv2.line(img,(bb1[0][0],bb1[0][1]),(bb2[0][0], bb2[0][1]),(0,0,0),5)
            group_boxes.append(box)
        else:
            other_boxes.append(box)

    return img, group_boxes, other_boxes

#ボックスのグループ化を行う
#注意:vctは使ってないので，使いたいときはちゃんとmake_vctを編集してね
def box_group_last(img, boxes, vct):
    # print("box_group_last")
    groups = []
    others = boxes
    while others != []:
        stash = others[0]
        img, group, others = original_nms(img, others[0], others[1:])
        flag = True
        i = 0
        num = len(group)
        while i < num:
            img, g, others = original_nms(img, group[i], others)
            count = 0
            if g != []:
                group.extend(g)
                num += len(g)
            i += 1
        # if group != []:
        #     group.append(stash)
        # else:
        #     print("stash : " + str(stash))
        #     others.append(stash)
        group.append(stash)
        if group != []:
            groups.append(group)
    return img, groups


def box_group_old(img, boxes, vct):
    lens = len(vct)
    strs = 0
    cnt_box = [] #類似してるボックスのインデックスたち
    box_name = [0]
    #類似度を計算して、近いベクトルを保存するよ
    thres = 1000
    box_index = []
    for i in range(lens):
        buf = []
        bb1 = ((boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), boxes[i][4])
        bb1_hist_b, bb1_hist_g, bb1_hist_r = rect_color_hist(img, bb1)
        cv2.putText(img, text=str(i), org=(bb1[0]),
                        fontFace=3,
                        fontScale=0.5,
                        color=(255, 255, 255))
        if i not in box_name:
            buf.append(bb1)
            yoko = hanbun(bb1)[0]
            yoko_min = bb1[0][0] - yoko
            yoko_max = bb1[0][0] + yoko
            vector = vct[0]
            index = vct[1]
            box_name.append(i)
            for j in range(strs,lens):
                if j not in box_name:
                    bb2 = ((boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), boxes[j][4])
                    bb2_hist_b, bb2_hist_g, bb2_hist_r = rect_color_hist(img, bb2)
                    print("-------------------------")
                    print("bb1 : " + str(bb1))
                    print("bb2 : " + str(bb2))
                    # print(bb1_hist_b)
                    # print(bb1_hist_g)
                    # print(bb1_hist_r)
                    # print(bb2_hist_b)
                    # print(bb2_hist_g)
                    # print(bb2_hist_r)

                    yoko2 = hanbun(bb2)[0]
                    yoko2_min = bb2[0][0] - yoko2
                    yoko2_max = bb2[0][0] + yoko2
                    if yoko_max <= yoko2_max:
                        diff = yoko_max - yoko2_min
                    else:
                        diff = yoko2_max - yoko_min
                    mama = yoko_max - yoko_min

                    #色
                    ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, 0)
                    ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, 0)
                    ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, 0)
                    print(ret_b)
                    print(ret_g)
                    print(ret_r)
                    print("------------------------")

                    if False:
                        print("--------------------------")
                        print("bb1 : " + str(bb1))
                        print("bb2 : " + str(bb2))
                        print("yoko_min : " + str(yoko_min))
                        print("yoko_max : " + str(yoko_max))
                        print("yoko2_min : " + str(yoko2_min))
                        print("yoko2_max : " + str(yoko2_max))
                    # scr = cos_sim(vector[0], vct[j][0])
                    # print(scr)
                    # if scr < thres:
                    # if yoko_min < yoko2_max or yoko_max > yoko2_min:
                    # if diff/mama > mama/2:
                    if diff > 0 and ret_b > 0 and ret_g > 0 and ret_r > 0:
                    # if ret_b > 0 and ret_g > 0 and ret_r > 0:
                        box_name.append(j)
                        # bb = ((boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), boxes[j][4])
                        buf.append(bb2)
            cnt_box.append(buf)
        strs = strs + 1
    return cnt_box


def sort_boxesAndLabels(boxes, labels):
    boxes = np.array(boxes)
    # print(boxes.shape)
    # print(boxes)
    # box_cnt = np.array(boxes[:,0])
    box_cnt = np.array(boxes[:,1])
    # print(box_cnt)
    # # print(box_cnt.shape)
    # box_x = []
    # for i in range(len(box_cnt)):
    #     box_x.append(box_cnt[i][0])
    sorted_boxes = boxes[np.argsort(box_cnt)]
    sorted_labels = labels[np.argsort(box_cnt)]
    return sorted_boxes, sorted_labels


def sort_boxesAndLabels2(boxes):
    boxes = np.array(boxes)
    # print(boxes.shape)
    # print(boxes)
    # box_cnt = np.array(boxes[:,0])
    # boxes = boxes[:][0]
    # print(boxes)
    box_cnt = np.array(boxes[:,1])
    # print(box_cnt)
    # # print(box_cnt.shape)
    # box_x = []
    # for i in range(len(box_cnt)):
    #     box_x.append(box_cnt[i][0])
    sorted_boxes = boxes[np.argsort(box_cnt)]
    return sorted_boxes


def original_nms_botu(img, rect, rects):
    group_boxes = []
    other_boxes = []
    bb1 = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    yoko, tate = hanbun(bb1)
    l = bb1[0][0] - yoko
    r = bb1[0][0] + yoko
    u = bb1[0][1] - tate
    d = bb1[0][1] + tate

    print("===================")
    print("===================")
    print("rect : " + str(rect))
    for box in rects:
        print("-------------------")
        print("box : " + str(box))
        bb2 = ((box[0], box[1]), (box[2], box[3]), box[4])
        yoko2, tate2 = hanbun(bb2)
        l2 = bb2[0][0] - yoko2
        r2 = bb2[0][0] + yoko2
        u2 = bb2[0][1] - tate2
        d2 = bb2[0][1] + tate2
        clear_x = True
        clear_y = True
        flag = True
        # if l < l2:
        #     if l2 >= r:
        #         clear_x = True
        # else:
        #     if l >= r2:
        #         clear_x = True
        # #d2の方が上
        # if d > d2:
        #     if u <= d2:
        #         clear_y = True
        # else:
        #     if u2 <= d:
        #         clear_y = True

        # if l > r2 or r < l2:
        #     flag = False
        # if u > d2 or u < d2:
        #     flag = False
        print(l,l2)
        print(r,r2)

        # print(u,u2)
        # print(d,d2)
        if l2 > r or r2 < l:
            clear_x = False
        if u2 > d or u2 < d:
            clear_y = False
        print(clear_x)
        print(clear_y)
        # if clear_x and clear_y:
        if clear_x and clear_y:
            img = cv2.line(img,(bb1[0][0],bb1[0][1]),(bb2[0][0], bb2[0][1]),(0,0,0),5)
            group_boxes.append(box)
        else:
            other_boxes.append(box)
    return img, group_boxes, other_boxes


def original_nms(img, rect, rects):
    group_boxes = []
    other_boxes = []
    # print(rect)
    bb1 = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    area_bb1 = bb1[1][0] * bb1[1][1]
    for box in rects:
        # print("-------------------")
        # print("box : " + str(box))
        bb2 = ((box[0], box[1]), (box[2], box[3]), box[4])
        area_bb2 = bb2[1][0] * bb2[1][1]
        yoko2, tate2 = hanbun(bb2)

        #bb1とbb2の共通する頂点を得る
        int_pts = cv2.rotatedRectangleIntersection(bb1, bb2)[1]
        inter = 0.0
        if int_pts is not None:
            #convexhull は　凸法を計算
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            #order_ptsの面積を計算
            int_area = cv2.contourArea(order_pts)
            inter = int_area * 1.0 / (area_bb1 + area_bb2 - int_area + cfgs.EPSILON)
        #ちょっとでも重なってたら結合ね
        if inter > 0:
            img = cv2.line(img,(bb1[0][0],bb1[0][1]),(bb2[0][0], bb2[0][1]),(0,0,0),5)
            group_boxes.append(box)
        else:
            other_boxes.append(box)
    return img, group_boxes, other_boxes


def rect_change(rect):
    return [int(rect[0][0]), int(rect[0][1]), int(rect[1][0]), int(rect[1][1]), int(rect[2])]


def extend_rects(rects):
    return_rects = []
    if EXTEND_RECD:
        for rect in rects:
            if rect[1][0] > rect[1][1]:
                if abs(rect[2]) >= 45:
                    a = ((rect[0][0], rect[0][1]), (rect[1][0]+RECT_SIZE, rect[1][1]), rect[2])
                else:
                    a = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]+RECT_SIZE), rect[2])
            else:
                if abs(rect[2]) > 45:
                    a = ((rect[0][0], rect[0][1]), (rect[1][0]*RECT_SIZE, rect[1][1]), rect[2])
                else:
                    a = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]+RECT_SIZE), rect[2])

            return_rects.append(a)
        return return_rects
    else:
        return rects


def grouping_by_nms(img, rects):
    vct = []
    # print(rects)
    img, cnt_box = box_group_last(img, rects, vct)
    final_rects = []
    for cnt_box_part in cnt_box:
        # print(len(cnt_box_part))
        # print(cnt_box_part)
        # connect_rect_new(img, cnt_box_part, verbose = False)
        points = []
        flag_theta = 0
        for rect in cnt_box_part:
            if rect[2] == 0 or abs(rect[2]) == 90:
                pass
            elif rect[2] > 0:
                flag_theta += 1
            else:
                flag_theta -= 1
            # print(rect)
            rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
            points.extend(calc_4points(img, rect))
            # points = calc_4points(rect)
        points = np.array(points)
        if False:
            print("============")
            print(points)
            print(points.shape)
            print(points.dtype)
            print(points[0].dtype)
            print(points[0][0].dtype)
            print(points[0][0][0].dtype)

        # rect = cv2.minAreaRect(np.array(points))
        rect = cv2.minAreaRect(points)
        # img, checked_rect = check_rect(img, rect)
        final_rects.append(rect)
        if flag_theta == 0:
            pass
        elif flag_theta * rect[2] < 0:
            rect = convert_rect(rect)
    final_rects = extend_rects(final_rects)
    return img, final_rects


def draw_rotate_box_cv_cont(img, boxes, boxes_axis, labels, scores, img_name):
    img_ori = img + np.array(cfgs.PIXEL_MEAN)
    save_dir = os.path.join("./rect_data", "original",)
    cv2.imwrite(save_dir + '/' + "original_" + img_name,
                img_ori)
    # print("img_ori : " + str(img_ori.shape))
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    boxes_axis = boxes_axis.astype(np.int64)
    labels = labels.astype(np.int32)
    boxes, labels = sort_boxesAndLabels(boxes, labels)
    # img = cv2.line(img,(0, 924),(719, 924),(0,0,255),5)
    # img = cv2.line(img,(0, IMG_LOW),(719, IMG_LOW),(0,255,0),5)

    # img = np.array(img, np.float32)
    # img = np.array(img*255/np.max(img), np.uint8)

    #ぼかすお
    #バイラテラルフィルタ
    # img = cv2.bilateralFilter(img,9,75,75)
    #平均
    # img = cv2.blur(img,(5,5))
    #ガウシアンフィルタ
    if GAUSS_FLAG:
        gauss = 5
        img = cv2.GaussianBlur(img,(gauss,gauss),0)
        # img = cv2.GaussianBlur(img,(gauss,gauss),0)
        # img = cv2.GaussianBlur(img,(gauss,gauss),0)
        # img = cv2.GaussianBlur(img,(gauss,gauss),0)

    for i in range(len(boxes)):
        box = boxes[i]
        # print("---------------------")
        # print(boxes[i])
        img, boxes[i] = check_rect(img, box)
        # print(boxes[i])
        # if box[2] < box[3] and box[4] > 0 and box[4] < 45:
        #     boxes[i] = convert_rect_origin(box)
        # elif box[2] > box[3] and box[4] < 0 and box[4] > -45:
        #     boxes[i] = convert_rect_origin(box)

    num_of_objects = 0

    boxes, vct = make_vct(boxes, labels)
    img_ori, cnt_box = box_group(img_ori, boxes, vct)
    # print("img_ori2 : " + str(img_ori.shape))
    # print("-------------")
    # print("cnt_box : " + str(np.array(cnt_box).shape))
    count = 0
    for i in range(len(cnt_box)):
        for j in range(len(cnt_box[i])):
            # print(cnt_box[i][j])
            cnt_box[i][j] = ((cnt_box[i][j][0], cnt_box[i][j][1]), (cnt_box[i][j][2], cnt_box[i][j][3]), cnt_box[i][j][4])
            box = cv2.boxPoints(cnt_box[i][j])
            box = np.int0(box)
            img_ori = cv2.drawContours(img_ori,[box],-1,green,2)
            # print("img_ori3 : " + str(img_ori.shape))
            # cv2.putText(img, text=str(i) + "," + str(count), org=(cnt_box[i][j][0]), fontFace=3, fontScale=0.5, color=(255,255,255))
            # cv2.putText(img, text=str(i) + "," + str(count) + "," + str(cnt_box[i][j]), org=(cnt_box[i][j][0]), fontFace=3, fontScale=0.5, color=(255,255,255))
            # print("----------------")
            # print(str(i) + "," + str(count))
            # print(cnt_box[i][j])
            # cv2.putText(img, text=str(cnt_box[i][j][0]), org=(cnt_box[i][j][0]), fontFace=3, fontScale=0.5, color=(255,255,255))
            count += 1
    all_rect = []

    #cnt_boxの中に入ってるボックスを結合したい
    #新アイデア：各ボックスを構成する直線群をすべて取得し、それらの交点をすべて計算する。
    #んで、一番上、左、右、下の点を取得してそれでボックスを構成しましょう。
    for cnt_box_part in cnt_box:
        # print(len(cnt_box_part))
        # print(cnt_box_part)
        # connect_rect_new(img, cnt_box_part, verbose = False)
        points = []
        flag_theta = 0
        for rect in cnt_box_part:
            if rect[2] == 0 or abs(rect[2]) == 90:
                pass
            elif rect[2] > 0:
                flag_theta += 1
            else:
                flag_theta -= 1
            # print(rect)
            points.extend(calc_4points(img, rect))
            # points = calc_4points(rect)
        points = np.array(points)
        if False:
            print("============")
            print(points)
            print(points.shape)
            print(points.dtype)
            print(points[0].dtype)
            print(points[0][0].dtype)
            print(points[0][0][0].dtype)

        # rect = cv2.minAreaRect(np.array(points))
        rect = cv2.minAreaRect(points)
        if flag_theta == 0:
            pass
        elif flag_theta * rect[2] < 0:
            rect = convert_rect(rect)
        all_rect.append(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img_ori = cv2.drawContours(img_ori,[box],-1,blue,2)
        # print("img_ori4 : " + str(img_ori.shape))

    #でっかいrectをnmsしたい

    all_rect2 = []
    for rect in all_rect:
        cv2.putText(img, text=str(int(rect[0][0]))+","+str(int(rect[0][1])), org=((int(rect[0][0]), int(rect[0][1]))), fontFace=3, fontScale=0.5, color=(255,255,255))
        all_rect2.append(rect_change(rect)) # cv2.imwrite(save_dir + '/' + a_img_name + '_original.jpg',


    all_rect2 = sort_boxesAndLabels2(all_rect2)
    for i in range(len(all_rect2)):
        box = all_rect2[i]
        img, all_rect2[i] = check_rect(img, box)

    final_rects = all_rect2
    rects_num = len(final_rects)
    flag = True
    # while flag == 2:
    while flag:
        img_ori, final_rects = grouping_by_nms(img_ori, final_rects)
        # print("img_ori5 : " + str(img_ori.shape))
        if len(final_rects) == rects_num:
        # if True:
            flag = False
            # flag = 0
        else:
            flag = True
            # flag += 1
            for i in range(len(final_rects)):
                final_rects[i] = rect_change(final_rects[i])
            rects_num = len(final_rects)

    #ちっさいrectはオワオワリ
    #final_rectsは((cnt), (size), angle)
    del_flag = True
    if del_flag:
        del_list = []
        for i in range(len(final_rects)):
                # print(final_rects[i][1])
                if final_rects[i][1][0] < 100 and final_rects[i][1][1] < 100:
                    # print("yeah")
                    del_list.append(final_rects[i])
        for i in del_list:
            if i in final_rects:
                final_rects.remove(i)
        print("--------final_rects---------")
        print(final_rects)

    print("num of rects : " + str(len(final_rects)))
    for rect in final_rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img_ori = cv2.drawContours(img_ori,[box],-1,red,2)
        # print("img_ori6 : " + str(img_ori.shape))


    path_w = './rect_data/text/' + img_name + '.txt'
    with open(path_w, mode='w') as f:
        for box in final_rects:
            s = str(int(box[0][0])) + "," + str(int(box[0][1])) + "," + str(int(box[1][0])) + "," + str(int(box[1][1])) + "," + str(int(box[2])) + "\n"
            s = str(box[0][0]) + "," + str(box[0][1]) + "," + str(box[1][0]) + "," + str(box[1][1]) + "," + str(box[2]) + "\n"
            f.write(s)

    # cv2.putText(img,
    #             text=str(num_of_objects),
    #             org=((img.shape[1]) // 2, (img.shape[0]) // 2),
    #             fontFace=3,
    #             fontScale=1,
    #             color=(255, 0, 0))

    return img_ori


if __name__ == '__main__':
    img_array = cv2.imread("/home/yjr/PycharmProjects/FPN_TF/tools/inference_image/2.jpg")
    img_array = np.array(img_array, np.float32) - np.array(cfgs.PIXEL_MEAN)
    boxes = np.array(
        [[200, 200, 500, 500],
         [300, 300, 400, 400],
         [200, 200, 400, 400]]
    )

    # test only draw boxes
    labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES
    scores = np.zeros_like(labes)
    imm = draw_boxes_with_label_and_scores(img_array, boxes, labes ,scores)
    # imm = np.array(imm)

    cv2.imshow("te", imm)

    # test only draw scores
    labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES_WITH_SCORES
    scores = np.random.rand((len(boxes))) * 10
    imm2 = draw_boxes_with_label_and_scores(img_array, boxes, labes, scores)

    cv2.imshow("te2", imm2)
    # test draw label and scores

    labels = np.arange(1, 4)
    imm3 = draw_boxes_with_label_and_scores(img_array, boxes, labels, scores)
    cv2.imshow("te3", imm3)

    cv2.waitKey(0)
