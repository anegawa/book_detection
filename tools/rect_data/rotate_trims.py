from PIL import Image
import os, sys
import cv2
import numpy as np


# img_files = os.listdir("")
dirs = os.listdir("./trims/")
for i, dir_name in enumerate(dirs):
    print("------------------------------")
    print(dir_name)
    a = os.path.join("./trims", dir_name)
    files = os.listdir(a)
    for j, file_name in enumerate(files):
        a_file = os.path.join("./trims" ,dir_name, file_name)
        print("file_name : " + str(a_file))
        img = cv2.imread(a_file)
        h, w = img.shape[:2]
        center = (w/2, h/2)
        angle = 90
        angle_rad = angle/180.0*np.pi

        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
        img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
        if angle == 90:
            op = os.path.join("./rotate_trim", dir_name)
        elif angle == -90:
            op = os.path.join("./rotate_trim2", dir_name)
        if not os.path.exists(op):
            os.mkdir(op)
        cv2.imwrite(op + "/" + file_name, img_rot)
