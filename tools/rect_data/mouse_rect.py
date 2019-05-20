# -*- coding: utf-8 -*-

import cv2, os

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


if __name__ == "__main__":
    #入力画像
    dir = "./original/"
    dirs = os.listdir(dir)
    for i, file_name in enumerate(dirs):
        print(file_name)
        a_file = os.path.join(dir, file_name)
        read = cv2.imread(a_file)

        #表示するWindow名
        window_name = file_name

        #画像の表示
        cv2.imshow(window_name, read)

        #コールバックの設定
        mouseData = mouseParam(window_name)

        # op = os.path.join("./original", file_name)
        # if not os.path.exists(op):
        #     os.mkdir(op)

        s = []
        while 1:
            cv2.waitKey(20)
            #左クリックがあったら表示
            if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
                # img = cv2.line(read,(),(int(a),950),(0,255,0),5)
                firstpoint = mouseData.getPos()
                s.append(mouseData.getPos())
                # print(mouseData.getPos())

            # elif mouseData.getEvent() == cv2.EVENT_LBUTTONUP:
            #     cv2.line(read, firstpoint, endpoint, (0, 255, 0), 5)
            #     endpoint = mouseData.getPos()
            #
            # elif mouseData.getEvent() == cv2.EVENT_LBUTTONDBCLK:
            #     s.pop(-1)
            # elif mouseData.getEvent() == cv2.EVENT_LBUTTONDBCLK:
            #


            #右クリックがあったら終了
            elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
                break;

        w_path = os.path.join("./labels", file_name + ".txt")
        with open(w_path, mode='w') as f:
            for string in s:
                f.write(str(string))

        cv2.destroyAllWindows()
        print("Finished")
