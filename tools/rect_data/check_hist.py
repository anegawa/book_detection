import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def color_hist(img):
    hist_b = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_r = cv2.calcHist([img],[2],None,[256],[0,256])
    # normal = np.array(img_rot_trim).shape[0] * np.array(img_rot_trim).shape[1] / 30000
    # hist_b = np.array(hist_b) / normal
    # hist_g = np.array(hist_g) / normal
    # hist_r = np.array(hist_r) / normal
    return hist_b, hist_g, hist_r

if __name__ == '__main__':
    trim = "trims"
    directri = "./histgrams/original/"
    hist_data = []
    dirs2 = []
    dirs = os.listdir("./" + trim + "/")
    for i, dir_name in enumerate(dirs):
        print("------------------------------")
        print(dir_name)
        a = os.path.join("./" + trim, dir_name)
        files = os.listdir(a)
        buf = []
        dirs2.append(dir_name)
        # for j, file_name in enumerate(files):
        for j in range(len(files)):
            # file_name = dir_name + "_" + str(j) + ".jpg"
            file_name = str(j) + ".jpg"
            a_file = os.path.join("./" + trim ,dir_name, file_name)
            if os.path.exists(a_file):
                # print("\t" + str(j) + " : " + str(file_name))
                print(str(j) + " : " + str(file_name))
                img = cv2.imread(a_file)
                histgram = color_hist(img)
                normal = np.array(img).shape[0] * np.array(img).shape[1] / 30000
                histgram_norm = np.array(histgram) / normal
                plt.figure()
                for i in range(3):
                    plt.subplot(3,1,i+1)
                    # plt.plot(histgram_norm[i]);
                    plt.plot(histgram[i]);
                    plt.savefig(directri + dir_name + "_" + file_name)

                    # sum = 0
                    # for j in range(len(histgram_norm[i])):
                    #     sum += histgram_norm[i][j]
                    # print("sum" + str(i) + " : " + str(sum))
                    # print("max" + str(i) + " : " + str(max(histgram_norm[i])))
