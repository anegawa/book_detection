import cv2
import os
import numpy as np
import math

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
    trim = "trims"
    hist_data = []
    dirs2 = []
    dirs = os.listdir("./" + trim + "/")
    for i, dir_name in enumerate(dirs):
        print("------------------------------")
        print("dir_name : " + dir_name)
        op = os.path.join("./hough", dir_name)
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
                # print(np.array(img).shape)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                edges = cv2.Canny(gray,0,500,apertureSize = 5)
                kakuritu = False
                OK_lines = []
                if kakuritu:
                    d = "hough_kakuritu"
                    minLineLength = 100
                    maxLineGap = 10
                    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
                    print(lines)
                    if lines is not None:
                        for i in range(len(lines)):
                            for x1,y1,x2,y2 in lines[i]:
                                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.imwrite("./" + d + "/" + dir_name + "_" + str(j) + "_hough.jpg",img)
                    else:
                        print(lines)

                else:
                    d = "hough"
                    lines = cv2.HoughLines(edges,1,np.pi/180,200)
                    print("lines : " + str(np.array(lines).shape))

                    if lines is not None:
                        for i in range(len(lines)):
                            flag = False
                            for rho,theta in lines[i]:
                                kodo = theta/math.pi*180
                                # print("lines[i] : " + str(lines[i]))
                                # if theta < math.pi/2 or theta > 3*math.pi/2:

                                if kodo < 45 or kodo > 135:
                                # if True:
                                    print("theta : " + str(theta/math.pi*180))
                                    a1 = np.cos(theta)
                                    b1 = np.sin(theta)
                                    x0 = a1*rho
                                    y0 = b1*rho
                                    x1 = int(x0 + 1000*(-b1))
                                    y1 = int(y0 + 1000*(a1))
                                    x2 = int(x0 - 1000*(-b1))
                                    y2 = int(y0 - 1000*(a1))
                                    # flag = True
                            # if flag:

                                    a, b = calc_equ([x1, y1], [x2, y2])
                                    print("point : " + str([x1, x2]) + str([x2,y2]))
                                    black_count = 0
                                    max_black_count = 0
                                    for i in range(np.array(img).shape[1]):
                                        if a != 100000:
                                            y = a*i + b
                                        else:
                                            i = b
                                            #print("i , y : " + str([i, y]))
                                        if y > 0 and y < np.array(img).shape[0]:
                                            print("black_count : " + str(black_count))
                                            #print("i , y : " + str([i, y]))
                                            iro = img[int(y)][i]
                                            print("color : " + str(iro))
                                            if iro[0] < 200 and iro[1] < 200 and iro[2] < 200:
                                                black_count += 1
                                            else:
                                                if max_black_count < black_count:
                                                    max_black_count = black_count
                                                black_count = 0
                                    print("max_black_count : " + str(max_black_count))
                                    if max_black_count >= 5:
                                        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                                        OK_lines.append([(x1, y1),(x2, y2), kodo])
                                        print("OK")
                        # for i in OK_lines:
                        #     print(i)
                        for i in OK_lines:
                            print(i)
                            cv2.line(img,i[0],i[1],(0,0,255),2)

                    else:
                        print(lines)

                    cv2.imwrite("./" + op + "/" + dir_name + "_" + str(j) + "_hough.jpg",img)
