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


def normalize_point(points):
    for i in range(len(points)):
        # points[i] = np.int32(np.points[i])
        points[i] = np.array(points[i])
    points = [np.array([points[0]]), np.array([points[1]]), np.array([points[2]]), np.array([points[3]])]
    return np.array(points)


def sort_boxes(boxes):
    # boxes = np.array(boxes)
    # # print(boxes)
    # box_cnt = np.array(boxes[:,0])
    # sorted_boxes = boxes[np.argsort(box_cnt)]
    # return sorted_boxes
    boxes = np.array(boxes)
    xs = []
    for i in boxes:
        xs.append(int(i[0][0]))
    xs = np.array(xs)
    # print(xs)
    # print(np.argsort(xs))
    sorted_boxes = boxes[np.argsort(xs)]
    # print(sorted_boxes)
    return sorted_boxes


def sort_boxes2(boxes):
    boxes = np.array(boxes)
    # print(boxes)
    box_cnt = np.array(boxes[:,0])
    sorted_boxes = boxes[np.argsort(box_cnt)]
    return sorted_boxes

if __name__ == '__main__':
    EPSILON = 1e-5
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 255, 0)
    dir = "./labels"
    text_dir = "./text"
    save_dir = "./test"
    dirs = os.listdir(dir)
    for i, file_name in enumerate(dirs):
        print("------------------------------")
        img_file_name = file_name.strip().split('.txt')[0]
        text_file_name = file_name.strip().split('original_')[1]
        print(file_name)
        # print(img_file_name)
        labels_path = os.path.join(dir, file_name)
        text_path = os.path.join(text_dir, text_file_name)
        b = os.path.join("./original", img_file_name)
        img = cv2.imread(b)
        with open(labels_path, mode="r") as f:
            labels_s = f.read()

        #textの箱が完成
        labels_rects = []
        with open(text_path, mode="r") as f:
            for line in f:
                line_split = line.strip().split(',')
                [x_c, y_c, w, h, theta] = line_split
                line_split = [x_c, y_c, w, h, theta]
                # line_split = [np.int0(x_c), np.int0(y_c), np.int0(w), np.int(h), np.int0(theta)]
                labels_rects.append(line_split)

        #labelの箱を作る
        ss = labels_s.strip().split('(')
        color_flag = 0
        text_rects = []
        text_point = []
        points = []
        point = []
        for si in ss:
            # print("color_flag : " + str(color_flag))
            si2 = si.strip().split(')')[0]
            si3 = si2.strip().split(',')
            # print("si3 : " + str(si3))
            if si3[0] is not '':
                color_flag += 1
                point.append((int(si3[0]), int(si3[1])))
            if color_flag == 4:
                # print("point : " + str(point))
                point = normalize_point(point)
                rect = cv2.minAreaRect(point)
                point = []
                text_rects.append(rect)
                color_flag = 0

        text_rects = sort_boxes(text_rects)

        print("labels_rects")
        for p in range(len(labels_rects)):
            label_rect = labels_rects[p]
            label_rect = [np.float32(y) for y in label_rect]
            # print(label_rect)
            label_rect = [np.int0(y) for y in label_rect]
            labels_rects[p] = label_rect
            # print(labels_rects[p])
            # print(a)
        labels_rects = sort_boxes2(labels_rects)
        # print(text)
        # print("iou-----------------------------------------------------")

        #iouの計算
        for p in range(len(labels_rects)):
            label_rect = labels_rects[p]
            label_rect = [np.float32(y) for y in label_rect]
            label_rect = [np.int0(y) for y in label_rect]
            bb1 = ((label_rect[0], label_rect[1]), (label_rect[2], label_rect[3]), label_rect[4])
            box = cv2.boxPoints(bb1)
            box = np.int0(box)
            img = cv2.drawContours(img,[box],-1,blue,2)
            cv2.putText(img, text=str(p), org=(bb1[0]), fontFace=3, fontScale=0.5, color=blue)
            print("bb1 : " + str(p) + str(bb1))
            # print(p)
            # print(len(text_rects))

            if p < len(text_rects):
                # sys.exit(1)
                bb2 = text_rects[p]
            # print(bb1)
                bb2 = [np.int0(y) for y in bb2]
                bb2[0] = tuple(bb2[0])
                bb2[1] = tuple(bb2[1])
                bb2 = tuple(bb2)
                box = cv2.boxPoints(bb2)
                box = np.int0(box)
                img = cv2.drawContours(img,[box],-1,green,2)
                cv2.putText(img, text=str(p), org=(bb2[0]), fontFace=3, fontScale=0.5, color=green)

            # print(bb2)
            # bb1 = [np.float32(i) for i in bb1]
            # bb1 = tuple(bb1)
            # print(bb1)
            # # bb1 = cv2.boxPoints(bb1)
            # bb1 = np.int0(bb1)





                print("bb2 : " + str(p) + str(bb2))

                cnt_norm = np.linalg.norm(np.array(bb1[0]) - np.array(bb2[0]))
                print("cnt_norm : " + str(cnt_norm))

                area_bb1 = bb1[1][0] * bb1[1][1]
                area_bb2 = bb2[1][0] * bb2[1][1]

                print("area_diff : " + str(abs(area_bb1 - area_bb2)))

                int_pts = cv2.rotatedRectangleIntersection(bb1, bb2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area = cv2.contourArea(order_pts)
                    inter = int_area * 1.0 / (area_bb1 + area_bb2 - int_area + EPSILON)
                    print("inter : " + str(inter))
        save_path = os.path.join(save_dir, img_file_name)
        print("save_path : " + save_path)
        cv2.imwrite(save_path, img)
