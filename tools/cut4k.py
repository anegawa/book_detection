import cv2, os

if __name__ == '__main__':
    trim = "4ks"
    save_dir = "cut4ks"
    dirs = os.listdir("./" + trim + "/")
    for i, name4k in enumerate(dirs):
        a = os.path.join("./" + trim, name4k)
        im = cv2.imread(a)
        # print(int(im.shape[1]/5))
        av = int(im.shape[1]/7)
        av2 = int(im.shape[0]/4)
        ran = 0
        ran2 = 0
        # i = 0
        count = 0
        for j in range(7):
            ran2 = 0
            for k in range(4):
                # print(ran)
                # print(ran+av)
                dst = im[ran2:ran2+av2,ran:ran+av]
                ran2 = ran2 + av2
                cv2.imwrite("cut4ks/" + name4k + "_" + str(count) + '.jpg',dst)
                count += 1
            ran = ran + av
