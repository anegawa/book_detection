import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert
from PIL import Image

count = 0
def sort_boxes(boxes):
    boxes = np.array(boxes)
    # print(boxes)
    box_cnt = np.array(boxes[:,0])
    sorted_boxes = boxes[np.argsort(box_cnt)]
    return sorted_boxes


def trim_rotateRect(img, rect, dir_name):
    # print("anegawa")
    global count
    #回転変換行列の計算
    p = "./rect_data/trims_sp/"
    print(rect[0])
    print(rect[1])
    op1 = os.path.join(p, dir_name + "_" + str(int(float(rect[0]))) + "_" + str(int(float(rect[1]))) + "_" + str(int(float(rect[4]))) + ".jpg")
    # print(op1)
    # z_m = np.zeros(img.shape)
    # z_m[int(float(rect[0]))][int(float(rect[1]))] = 1
    rect = [np.float32(i) for i in rect]
    img = np.array(img)
    # print(img.shape)
    # print("==========")
    # print("img  : " + str(img.shape))
    print("rect : " + str(rect))
    # print("\trect : " + str(rect))
    ##########################################################################################
    print(np.array(img).shape)
    h, w = np.array(img).shape[:2]
    size = (w, h)

    # 回転角の指定
    angle = rect[4]
    angle_rad = angle/180.0*np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    center = (rect[0], rect[1])
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)
    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()

    affine_matrix[0][2] = affine_matrix[0][2] - rect[0] + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] - rect[1] + h_rot/2
    # print(affine_matrix)
    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    # z_m = cv2.warpAffine(z_m, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    ##########################################################################################

    # h, w = img.shape[:2]
    # size = (w, h)
    # angle = 0
    # center = (w/2, h/2)
    # scale = 1.0
    # rot = np.array([affine_matrix[0][0:2], affine_matrix[1][0:2]])
    # print(rot.shape)
    # print(rot)
    # x, y = np.dot([rect[0], rect[1]], rot) + [affine_matrix[0][2], affine_matrix[1][2]]
    # x,y = cv2.warpAffine((rect[0], rect[1]), affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    # x,y = np.where(z_m==1)
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # x = int(rect[0] * abs(math.cos(math.radians(rect[4]))) + rect[1]*abs(math.cos(math.cos(math.radians(90-rect[4])))))
    # y = int(rect[0] * abs(math.sin(math.radians(rect[4]))) + rect[1]*abs(math.sin(math.cos(math.radians(90-rect[4])))))

    # print(int(x), int(y))
    # cv2.circle(img_rot, (int(x),int(y)), 15, (255,255,255), thickness=3)
    #
    # affine_matrix = rotation_matrix.copy()
    # affine_matrix[0][2] = affine_matrix[0][2] - x + w_rot/2
    # affine_matrix[1][2] = affine_matrix[1][2] - y + h_rot/2

    # img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    # img_rot_trim = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.BORDER_WRAP)

    # rotation_matrix = cv2.getRotationMatrix2D((rect[0],rect[1]), rect[4], 1.0)
    # rotation_matrix = cv2.getRotationMatrix2D((w_rot/2, h_rot/2), rect[4], 1.0)

    # size = tuple(np.array([img.shape[1], img.shape[0]]))
    # size = tuple(np.array([lo, lo]))
    # img_rot = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_LINEAR)
    # # img_rot = np.float32(img_rot)
    # # img_rot_trim = cv2.getRectSubPix(img_rot, tuple((rect[2], rect[3])), tuple((rect[0], rect[1])))
    img_rot_trim = cv2.getRectSubPix(img_rot, tuple((rect[2], rect[3])), tuple((w_rot/2, h_rot/2)))
    # cv2.imwrite(op1, img_rot)

    # return img_rot
    return img_rot_trim


#拾い物　つかてない
def cut_after_rot(src_img, deg, center, size):
    rot_mat = cv2.getRotationMatrix2D(center, deg, 1.0)
    src_h,src_w,_ = src_img.shape
    rot_img = cv2.warpAffine(src_img, rot_mat, (src_w,src_h))
    #スライスによって領域を切り出す
    return rot_img[center[1]-size[1]//2:center[1]+size[1]//2, \
                   center[0]-size[0]//2:center[0]+size[0]//2, :]

img_path = "./inference_image/"
img_path_ori = "./rect_data/original/"
# img_path_ori = "./inference_results/Bhattacharyya_20190131/"
rect_data_path = "./rect_data/text/"
print("")
print("rect_data_path : " + str(rect_data_path))
print("img_path : " + str(img_path))
rect_data_files = os.listdir(rect_data_path)
img_files = os.listdir(img_path)

print("-----------------------")
# for i, file_name in enumerate(img_files):
for i, file_name in enumerate(rect_data_files):
    all_rects = []
    print("==================")
    print("file_name : " + str(file_name))
    # file_name : hondana2.jpg.txt
    # file_name : hondana3.jpg.txt
    # file_name : hondana6.jpg.txt
    # file_name : hondana10.jpg.txt
    # file_name : hondana9.jpg.txt
    # file_name : hondana7.jpg.txt
    # file_name : hondana8.jpg.txt
    # file_name : hondana4.jpg.txt
    # file_name : hondana_eng.jpg.txt
    # file_name : hondana5.jpg.txt
    img_file_name = file_name.strip().split('.txt')[0]
    img_file_path = img_path_ori + "original_" + img_file_name
    # img_file_path = img_path_ori + img_file_name + "_r_cont.jpg"
    print("img_file_path : " + str(img_file_path))
    img = cv2.imread(img_file_path)
    file_path = rect_data_path + str(file_name)# + ".txt"
    print("file_path : " + str(file_path))
    with open(file_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            [x_c, y_c, w, h, theta] = line_split
            line_split = [x_c, y_c, w, h, theta]
            all_rects.append(line_split)
    if len(all_rects) != 0:
        sorted_rects = sort_boxes(all_rects)
        dir_name = img_file_name.split(".jpg")[0]
        # print("dir_name : " + str(dir_name))
        print("num of rect : " + str(sorted_rects.shape[0]))
        for j, rect in enumerate(sorted_rects):
            # print("\trect : " + str(rect))
            trimed_img = trim_rotateRect(img, rect, dir_name)
            # trimed_img = cut_after_rot(img, rect[4], (rect[0], rect[1]), (rect[2], rect[3]))
            # cv2.imshow('image',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            op = os.path.join("rect_data/trims", dir_name)
            if not os.path.exists(op):
                os.mkdir(op)
            op1 = os.path.join(op, str(j) + ".jpg")

            # size = tuple([trimed_img.shape[1], trimed_img.shape[0]])
            # size2 = tuple([trimed_img.shape[0], trimed_img.shape[1]])
            # nagaihou = max(size)
            # size = tuple([nagaihou, nagaihou])
            # center = tuple([int(size[0]/2), int(size[1]/2)])
            # # center = tuple([nagaihou, nagaihou])
            # angle = 90
            # scale = 1.0
            # rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            # trimed_img = cv2.warpAffine(trimed_img, rotation_matrix, size, flags=cv2.INTER_CUBIC)

            h, w = trimed_img.shape[:2]
            if h < w:
                size = (w, h)
                angle = 90
                angle_rad = angle/180.0*np.pi

                w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
                h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
                size_rot = (w_rot, h_rot)

                center = (w/2, h/2)
                scale = 1.0
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                affine_matrix = rotation_matrix.copy()
                affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
                affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
                trimed_img = cv2.warpAffine(trimed_img, affine_matrix, size_rot, flags=cv2.BORDER_WRAP)
            gray_flag = False
            if gray_flag:
                img_gray = cv2.cvtColor(trimed_img, cv2.COLOR_BGR2GRAY)
                # 二値変換
                thresh = 100
                max_pixel = 255
                ret, trimed_img = cv2.threshold(img_gray,
                                                thresh,
                                                max_pixel,
                                                cv2.THRESH_BINARY)
            cv2.imwrite(op1, trimed_img)
