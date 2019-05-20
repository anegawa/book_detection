import numpy as np
import cv2

WIDE = 0

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

if __name__ == '__main__':
    img_path = "../inference_image/hon.jpg"
    img = cv2.imread(img_path)
    # rect = ((), (), )
    points = [(71, 172),(158, 178),(81, 932),(12, 926)]
    points = normalize_point(points)
    rect = cv2.minAreaRect(points)
    rect = ((int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])), int(rect[2]))
    rect_big = ((int(rect[0][0]), int(rect[0][1])), (int(rect[1][0])+10, int(rect[1][1])+10), int(rect[2]))

    print(rect)
    print(rect_big)
    hist_b, hist_g, hist_r = rect_color_hist(img, rect)
    hist_big_b, hist_big_g, hist_big_r = rect_color_hist(img, rect_big)
    # print("hist_b : " + str(hist_b))
    # print("hist_g : " + str(hist_g))
    # print("hist_r : " + str(hist_r))
    # diff_check_b = np.mean(np.array(hist_big_b - hist_b))
    # diff_check_g = np.mean(np.array(hist_big_g - hist_g))
    # diff_check_r = np.mean(np.array(hist_big_r - hist_r))
    diff_check_b = np.mean(np.array(hist_big_b)) - np.mean(np.array(hist_b))
    diff_check_g = np.mean(np.array(hist_big_g)) - np.mean(np.array(hist_g))
    diff_check_r = np.mean(np.array(hist_big_r)) - np.mean(np.array(hist_r))
    diff_check_sum = abs(diff_check_g) + abs(diff_check_b) + abs(diff_check_r)
    print("diff_check_sum : " + str(diff_check_sum))
    hist_flag_b = diff_check_b / diff_check_sum
    hist_flag_g = diff_check_g / diff_check_sum
    hist_flag_r = diff_check_r / diff_check_sum
    print("hist_b : " + str(diff_check_b))
    print("hist_g : " + str(diff_check_g))
    print("hist_r : " + str(diff_check_r))
    print("hist_flag_b : " + str(hist_flag_b))
    print("hist_flag_g : " + str(hist_flag_g))
    print("hist_flag_r : " + str(hist_flag_r))
