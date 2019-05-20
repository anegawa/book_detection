import cv2, os

if __name__ == '__main__':
    save_dir = "4ks_result"
    trim = "inference_results/Bhattacharyya_20190214_cnt_tres_each_0.4_str0_1"
    b = ".jpg_r_cont.jpg"
    trim2 = "4ks"
    dirs = os.listdir("./" + trim2 + "/")
    for i, name4k in enumerate(dirs):
        a = os.path.join("./" + trim, name4k) + "_"
        a2 = name4k.strip().split('.JPG')[0]
        # print(img_file_name)
        # im = cv2.imread(a)
        # print(int(im.shape[1]/5))
        # av = int(im.shape[1]/7)
        # av2 = int(im.shape[0]/4)
        # ran = 0
        # ran2 = 0
        # i = 0
        # count = 0
        imgs = []
        for count in range(7):
            # print(a + str(count) + b)
            im1 = cv2.imread(a + str(count*4) + b)
            im2 = cv2.imread(a + str(count*4 + 1) + b)
            im3 = cv2.imread(a + str(count*4 + 2) + b)
            im4 = cv2.imread(a + str(count*4 + 3) + b)
            im_v = cv2.vconcat([im1, im2, im3, im4])
            imgs.append(im_v)
        # for j in range(7):
        im_h = cv2.hconcat(imgs)
        t = os.path.join(save_dir, a2 + "_result.jpg")
        cv2.imwrite(t, im_h)
