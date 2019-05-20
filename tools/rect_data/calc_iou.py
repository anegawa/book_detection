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



if __name__ == '__main__':
    EPSILON = 1e-5
    green = (0, 255, 0)
    red = (0, 0, 255)

    bb1 = ((, ), (, ), )
    bb2 = ((, ), (, ), )


    print("bb1 : " + str(p) + str(bb1))


    bb2 = [np.int0(y) for y in bb2]
    bb2[0] = tuple(bb2[0])
    bb2[1] = tuple(bb2[1])
    bb2 = tuple(bb2)

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
