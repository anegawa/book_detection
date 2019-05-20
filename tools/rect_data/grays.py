import cv2
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import sys
sys.setrecursionlimit(10000)

#y = ax + bのaとbを計算するだけ
def calc_equ(point1, point2):
    if point1[0] - point2[0] != 0:
	    a = (point1[1] - point2[1])/(point1[0] - point2[0])
	    b = point1[1] - a*point1[0]
	    return [a, b]
    else:
        return [100000, point1[0]]


if __name__ == '__main__':
    trim = "trims_made"
    trim = "rotate_trim2"
    hist_data = []
    dirs2 = []
    dirs = os.listdir("./" + trim + "/")
    for i, dir_name in enumerate(dirs):
        print("------------------------------")
        print("dir_name : " + dir_name)
        op = os.path.join("./rotate_gray2", dir_name)
        if not os.path.exists(op):
            os.mkdir(op)

        a = os.path.join("./" + trim, dir_name)
        files = os.listdir(a)
        dirs2.append(dir_name)
        for j in range(len(files)):
            file_name = str(j) + ".jpg"
            a_file = os.path.join("./" + trim ,dir_name, file_name)
            if os.path.exists(a_file):
                print("\t" + str(j) + " : " + str(file_name))
                img = cv2.imread(a_file)
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                hist_img = cv2.equalizeHist(gray_img)
                cv2.imwrite(op + "/" + file_name,hist_img)
