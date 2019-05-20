import math
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import itertools
import matplotlib.pyplot as plt

from PIL import Image
import os, sys
import cv2
from os import environ

import pyocr
import pyocr.builders
METHOD = int(environ.get('METHOD', 3))


def color_hist(img):
    hist_b = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_r = cv2.calcHist([img],[2],None,[256],[0,256])
    return hist_b, hist_g, hist_r


def box_grouping(img, a_box, boxes):
    group_boxes = []
    other_boxes = []
    #基準ボックスの準備(ヒストグラムとｘの範囲)
    bb1 = ((a_box[0], a_box[1]), (a_box[2], a_box[3]), a_box[4])
    bb1_hist_b, bb1_hist_g, bb1_hist_r = rect_color_hist(img, bb1)
    img, bb1_min, bb1_max = calc_x_range(img, a_box)
    criterion_vct = np.array([0,1])
    bb1_vct = np.array([bb1[0][0], bb1[0][1]])

    #boxes内のボックスひとつひとつと比較
    for box in boxes:
        bb2 = ((box[0], box[1]), (box[2], box[3]), box[4])
        bb2_hist_b, bb2_hist_g, bb2_hist_r = rect_color_hist(img, bb2)
        bb2_vct = np.array([bb2[0][0], bb2[0][1]])

        yoko, tate = hanbun(bb2)
        bb2_min = bb2[0][0] - yoko
        bb2_max = bb2[0][0] + yoko

        if bb1_min <= bb2_min and bb1_max >= bb2_max:
            diff = 1
        else:
            diff = -1
        ret_b = cv2.compareHist(bb1_hist_b, bb2_hist_b, METHOD)
        ret_g = cv2.compareHist(bb1_hist_g, bb2_hist_g, METHOD)
        ret_r = cv2.compareHist(bb1_hist_r, bb2_hist_r, METHOD)

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

        if diff > 0 and ret_b > THRESHOLD and ret_g > THRESHOLD and ret_r > THRESHOLD and deg_flag:
            img = cv2.line(img,(bb1[0][0],bb1[0][1]),(bb2[0][0], bb2[0][1]),(0,0,255),5)
            group_boxes.append(box)
        else:
            other_boxes.append(box)

    return img, group_boxes, other_boxes


def comp(hist1, hist2):
    # print(np.array(hist1).shape)
    # print(np.array(hist2).shape)
    hist1 = tuple(hist1)
    hist2 = tuple(hist2)
    ret_b = 1 - cv2.compareHist(hist1[0], hist2[0], METHOD)
    ret_g = 1 - cv2.compareHist(hist1[1], hist2[1], METHOD)
    ret_r = 1 - cv2.compareHist(hist1[2], hist2[2], METHOD)
    # print(ret_b)
    # print(ret_g)
    # print(ret_r)
    # return ret_b * ret_g * ret_r
    return ret_b + ret_g + ret_r


def match_book_hist(hist_data1, hist_data2, dirname1, dirname2):
    # for i in range(len(hist_data)):
    #     for j in range(1, len(hist_data)):
    len1 = len(hist_data1)
    len2 = len(hist_data2)
    group1 = range(len1)
    group2 = range(len1, len1+len2)
    node_color = ["b"] *len1
    node_color.extend(["r"] *len2)

    g = nx.Graph()
    g.add_nodes_from(group1, bipartite=1)
    g.add_nodes_from(group2, bipartite=0)

    vals = []
    # print("--------comp score--------")

    for (i,j) in itertools.product(group1, group2):
        # val = np.random.randint(1, 10)
        # print(np.array(hist_data1).shape)
        # print(np.array(hist_data2).shape)
        # print("--------comp score--------")
        # print(i, j-len1)
        # val = math.log(comp(hist_data1[i], hist_data2[j-len1]))

        #hist version
        val = comp(hist_data1[i], hist_data2[j-len1])

        # print(str(i) + "," + str(j-len1) + " : " + str(val))
        vals.append(val)
        g.add_edge(i, j, weight=val)

    # print(len(vals))

    A,B = bipartite.sets(g)
    pos = dict()
    pos.update((n,(1,i)) for i,n in enumerate(A))
    pos.update((n,(2,i)) for i,n in enumerate(B))

    edge_width = [ d['weight']*0.3 for (u,v,d) in g.edges(data=True)]

    nx.draw_networkx(g, pos, node_color=node_color)
    nx.draw_networkx_edges(g, pos, width=edge_width)

    d = nx.max_weight_matching(g)
    # plt.axis("off")
    # plt.show()

    # print("d : " + str(d))
    # print("----------------------------")
    print("=========================================================")
    print("-------------------- hist matching ----------------------")
    print("     " + dirname1 + "    |    " + dirname2 + "   |   score" )
    print("---------------------------------------------------------")
    thres = 0
    for i in d:
        if i[0] < i[1] and vals[len2*i[0] + (i[1]-len1)] > thres:
            # print(len1*i[0] + len2*(i[1]-len1))
            print("\t   " + str(i[0]) + " : " + str(i[1]-len1) + "\t|   " + str(vals[len2*i[0] + (i[1]-len1)]))
        elif i[1] < [0] and vals[len2*i[1] + (i[0]-len1)] > thres:
            # print(len1*i[1] + len2*(i[0]-len1))
            print("\t   " + str(i[1]) + " : " + str(i[0]-len1) + "\t|   " + str(vals[len2*i[1] + (i[0]-len1)]))
    # print("============================")
    return d


def match_book_AKAZE(AKAZE_data1, AKAZE_data2, dirname1, dirname2):
    # for i in range(len(hist_data)):
    #     for j in range(1, len(hist_data)):
    len1 = len(AKAZE_data1)
    len2 = len(AKAZE_data2)
    group1 = range(len1)
    group2 = range(len1, len1+len2)
    node_color = ["b"] *len1
    node_color.extend(["r"] *len2)

    g = nx.Graph()
    g.add_nodes_from(group1, bipartite=1)
    g.add_nodes_from(group2, bipartite=0)

    vals = []
    # print("--------comp score--------")

    for (i,j) in itertools.product(group1, group2):
        # val = np.random.randint(1, 10)
        # print(np.array(hist_data1).shape)
        # print(np.array(hist_data2).shape)
        # print("--------comp score--------")
        # print(i, j-len1)
        # val = math.log(comp(hist_data1[i], hist_data2[j-len1]))


        # try:
        matches = bf.match(AKAZE_data1[i], AKAZE_data2[j-len1])
        dist = [m.distance for m in matches]
        print("dist : " + str(dist))
        if len(dist) != 0:
            ret = sum(dist) / len(dist)
        else:
            ret = 0
        # except cv2.error:
        #     ret = 100000
        print("ret : " + str(ret))
        mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)


        #hist version
        # val = comp(hist_data1[i], hist_data2[j-len1])

        # print(str(i) + "," + str(j-len1) + " : " + str(val))
        vals.append(ret)
        g.add_edge(i, j, weight=ret)

    # print(len(vals))

    A,B = bipartite.sets(g)
    pos = dict()
    pos.update((n,(1,i)) for i,n in enumerate(A))
    pos.update((n,(2,i)) for i,n in enumerate(B))

    edge_width = [ d['weight']*0.3 for (u,v,d) in g.edges(data=True)]

    nx.draw_networkx(g, pos, node_color=node_color)
    nx.draw_networkx_edges(g, pos, width=edge_width)

    d = nx.max_weight_matching(g)
    # plt.axis("off")
    # plt.show()

    # print("d : " + str(d))
    # print("----------------------------")
    print("=========================================================")
    print("------------------- AKAZE matching -----------------------")
    print("     " + dirname1 + "    |    " + dirname2 + "   |   score" )
    print("---------------------------------------------------------")
    thres = 0
    for i in d:
        if i[0] < i[1] and vals[len2*i[0] + (i[1]-len1)] > thres:
            # print(len1*i[0] + len2*(i[1]-len1))
            print("\t   " + str(i[0]) + " : " + str(i[1]-len1) + "\t|   " + str(vals[len2*i[0] + (i[1]-len1)]))
        elif i[1] < [0] and vals[len2*i[1] + (i[0]-len1)] > thres:
            # print(len1*i[1] + len2*(i[0]-len1))
            print("\t   " + str(i[1]) + " : " + str(i[0]-len1) + "\t|   " + str(vals[len2*i[1] + (i[0]-len1)]))
    # print("============================")
    return d


if __name__ == '__main__':
    trim = "trims_made"
    trim = "trims2"
    hist_data = []
    AKAZE_data = []
    imgs = []
    dirs2 = []
    dirs = os.listdir("./" + trim + "/")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    feature_flag = False

    for i, dir_name in enumerate(dirs):
        print("------------------------------")
        print(dir_name)
        a = os.path.join("./" + trim, dir_name)
        files = os.listdir(a)
        # print(files)
        buf = []
        buf2 = []
        img_buf = []
        size = (46, 626)
        dirs2.append(dir_name)
        # for j, file_name in enumerate(files):
        for j in range(len(files)):
            # file_name = dir_name + "_" + str(j) + ".jpg"
            file_name = str(j) + ".jpg"
            a_file = os.path.join("./" + trim ,dir_name, file_name)
            # print(a_file)
            if os.path.exists(a_file):
                print("\t" + str(j) + " : " + str(file_name))
                img = cv2.imread(a_file)
                img = cv2.resize(img, size)
                histgram = color_hist(img)
                normal = np.array(img).shape[0] * np.array(img).shape[1] / 30000
                histgram = np.array(histgram) / normal
                buf.append(histgram)

                if feature_flag:
                    img_buf.append(img)
                    (target_kp, target_des) = detector.detectAndCompute(img, None)
                    print(target_des)
                    buf2.append(target_des)

        hist_data.append(buf)
        AKAZE_data.append(buf2)
        imgs.append(img_buf)

    d = []
    # print(np.array(hist_data).shape)
    for i in range(len(hist_data)-1):
        buf = match_book_hist(hist_data[i], hist_data[i+1], dirs2[i], dirs2[i+1])
        # buf = match_book_AKAZE(AKAZE_data[i], AKAZE_data[i+1], dirs2[i], dirs2[i+1])
        d.append(buf)
    print("=========================================================")


    if feature_flag:
        for i in range(len(AKAZE_data)-1):
            # buf = match_book_hist(hist_data[i], hist_data[i+1], dirs2[i], dirs2[i+1])
            buf = match_book_AKAZE(AKAZE_data[i], AKAZE_data[i+1], dirs2[i], dirs2[i+1])
            d.append(buf)
        print("=========================================================")


    # print(d)
    # print(np.array(hist_data).shape)
    # print(np.array(hist_data[0]).shape)


    #
    # img = cv2.imread('home.jpg',0)
    #
    # # create a mask
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # masked_img = cv2.bitwise_and(img,img,mask = mask)
    #
    # # Calculate histogram with mask and without mask
    # # Check third argument for mask
    # hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
    # hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
    #
    # plt.subplot(221), plt.imshow(img, 'gray')
    # plt.subplot(222), plt.imshow(mask,'gray')
    # plt.subplot(223), plt.imshow(masked_img, 'gray')
    # plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    # plt.xlim([0,256])
    #
    # plt.show()
