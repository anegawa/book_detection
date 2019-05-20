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
    green = (0, 255, 0)
    red = (0, 0, 255)
    dir = "./labels"
    dirs = os.listdir(dir)
    for i, file_name in enumerate(dirs):
        print("------------------------------")
        img_file_name = file_name.strip().split('.txt')[0]
        print(file_name)
        print(img_file_name)
        a = os.path.join(dir, file_name)
        b = os.path.join("./original", img_file_name)
        img = cv2.imread(b)
        with open(a, mode="r") as f:
            s = f.read()
        # print(s)
        # print(type(s))
        ss = s.strip().split('(')

        color_flag = 0
        for si in ss:
            print("color_flag : " + str(color_flag))
            si2 = si.strip().split(')')[0]
            si3 = si2.strip().split(',')
            print("si3 : " + str(si3))
            if si3[0] is not '':
                if color_flag > 3:
                    img = cv2.circle(img, (int(si3[0]), int(si3[1])), 6, green, -1)
                    color_flag += 1
                else:
                    img = cv2.circle(img, (int(si3[0]), int(si3[1])), 6, red, -1)
                    color_flag += 1
            if color_flag == 8:
                color_flag = 0
        cv2.imwrite("./test/" + img_file_name, img)
        
