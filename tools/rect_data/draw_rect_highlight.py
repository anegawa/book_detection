import sys
sys.path.append("../../")
import numpy as np
import cv2
import math
from math import *
from libs.configs import cfgs

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
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
pp = (255, 0, 255)
midpp = (147, 112, 219)
midpp = (255, 0, 120)
yellow = (255, 255, 0)
midyellow = (255, 120, 0)
black = (0,0,0)
white = (255,255,255)
WIDE = 0
EXTEND_RECD = False
RECT_SIZE = 20
puttext_flag = True
Nan = np.nan


IMG_LOW = 900
GAUSS_FLAG = True
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
    if len(rect) == 5:
        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    # print("rect")
    # print(rect)
    rect = ((rect[0][0], rect[0][1]), (rect[1][0]+WIDE, rect[1][1]+WIDE), rect[2])
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


def normalize_point(points):
    for i in range(len(points)):
        # points[i] = np.int32(np.points[i])
        points[i] = np.array(points[i])
    points = [np.array([points[0]]), np.array([points[1]]), np.array([points[2]]), np.array([points[3]])]
    return np.array(points)


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


def calc_book_range(img, rect):
    if len(rect) == 3:
        rect = [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]
    return_boxes = []
    for i in range(cfgs.STRIDE_NUM):
        # print(rect)
        img, rect = check_rect(img, rect)
        # print(rect)
        bb = ((rect[0] + (i-int(cfgs.STRIDE_NUM/2)) * cfgs.STRIDE , rect[1]), (rect[2], rect[3]), rect[4])
        if bb[2] > 0:
            flag = 1
        else:
            flag = -1
        x,y = bb[0][0], bb[0][1]
        yoko, tate = hanbun(bb)
        # print("tate : " + str(tate))
        # print("yoko : " + str(yoko))
        h2 = IMG_LOW-rect[1]


        thre = img.shape[0]
        # print("tres : " + str(thre))

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


        # print("cnt_x : " + str(cnt_x))
        # print("cnt_y : " + str(cnt_y))
        # print("line_h : " + str(line_h))
        # print("line_w : " + str(line_w))
        # print("line_norm : " + str(line_norm))
        # img = cv2.line(img, (0, int(cnt_y + line_h/2)), (800, int(cnt_y + line_h/2)), (255,0,0), 5)
        # img = cv2.circle(img,(int(cnt_x), int(cnt_y)), 6, (0,255,0), -1)

        box1 = ((cnt_x, cnt_y), (line_norm, bb[1][1] + cfgs.WIDE_RANGE), bb[2])
        # box1 = [cnt_x, cnt_y, bb[1][1], line_norm,bb[2]]
        # img, box1 = check_rect(img, box1)
        # box1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        if puttext_flag:
            box = cv2.boxPoints(box1)
            box = np.int0(box)
            if cfgs.DRAW_FLAG:
                img = cv2.drawContours(img,[box],-1,black,2)
        return_boxes.append(box1)
    img = cv2.line(img,(bb[0][0]+int(yoko),bb[0][1]-int(h1)),(int(cnt_x-line_w/2),IMG_LOW),(0,0,255),5)

    # img = cv2.line(img,(bb[0][0],bb[0][1]),(int(a),950),(0,255,0),5)
    return img, return_boxes


def calc_x_range(img, rect):
    # print(img.shape)
    if len(rect) == 3:
        rect = [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]
    x_min_maxs = []
    for i in range(cfgs.STRIDE_NUM):
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
    anegawa = yoko*tan(radians(abs(bb[2])))
    img = cv2.line(img,(bb[0][0]+int(yoko),bb[0][1]-int(anegawa)),(int(a),950),(0,0,255),5)
    return img, x_min_maxs


if __name__ == '__main__':
    img_path = "../inference_image/hon6.jpg"
    img_path = "./original/original_hon6.jpg"
    img_path = "./hon6_2.jpg"
    img_path = "./grouping.png"
    save_path = "./save.jpg"
    img = cv2.imread(img_path)
    # img = np.zeros((960, 720, 3), np.uint8)
    # for i in range(960):
    #     for j in range(720):
    #         img[i][j] = (255, 255, 255)
    # img.fill(255)
    # rect = ((), (), )
    points = [(71, 172),(158, 178),(81, 932),(12, 926)]
    points = normalize_point(points)
    rect = cv2.minAreaRect(points)
    rect = ((int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])), int(rect[2]))
    rect_big = ((int(rect[0][0]), int(rect[0][1])), (int(rect[1][0])+10, int(rect[1][1])+10), int(rect[2]))

    rect = ((537, 170), (18, 58), -76)
    rect = ((419, 264), (25, 43), -75)
    # rect = ((419, 264), (25, 43), -75)

    flag = False
    if flag:
        img, boxes = calc_book_range(img, rect)
        rect2 = boxes[0]
        # rect = [537, 170, 18, 58, -76]

        # print(rect)
        # print(rect_big)
        box = cv2.boxPoints(rect2)
        box = np.int0(box)
        #囲む
        img = cv2.drawContours(img,[box],-1,green,3)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # #囲む
        # img = cv2.drawContours(img,[box],-1,red,5)
        img = cv2.circle(img,(int(rect2[0][0]), int(rect2[0][1])), 9, (255,0,0), -1)
        img = cv2.circle(img,(rect[0][0], rect[0][1]), 9, (0,0,255), -1)
    else:
        img, min_maxs = calc_x_range(img, rect)
        x = (min_maxs[0][1]+min_maxs[0][0])/2
        x_size = min_maxs[0][1]-min_maxs[0][0]
        rect_big = ((x, img.shape[0]/2), (x_size, img.shape[0]), 0)
        box = cv2.boxPoints(rect_big)
        box = np.int0(box)
        # img = cv2.drawContours(img,[box],-1,green,3)
        # sys.exit()

    # img = cv2.drawContours(img,[box],-1,yellow,3)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #囲む
    img = cv2.drawContours(img,[box],-1,red,5)

    cv2.imwrite(save_path, img)
