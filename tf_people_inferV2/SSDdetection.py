import cv2 as cv
import datetime
import time


# 行人检测（目标检测 人数<100）
class People_detection:
    def __init__(self):
        # 导入及读取tf模型
        self.inference_pb = "./model/SSD_model/frozen_inference_graph.pb"
        self.graph_txt = "./model/SSD_model/graph.pbtxt"
        self.net = cv.dnn.readNetFromTensorflow(self.inference_pb, self.graph_txt)

    def detection(self, image, args):
        place_num = args.monitoring_place
        if place_num == '1':
            place_name = "tower"
        elif place_num == '2':
            place_name = "rock"
        elif place_num == '3':
            place_name = "ruins"
        else:
            place_name = "rock"
        # 获取当前时间，实现FPS的显示
        t_start = time.time()
        fps = 0

        # 人数初始化
        people_num = 0
        image = cv.flip(image, 1)

        # 获取视频宽高
        h, w = image.shape[:2]

        # 将图片导入模型进行处理
        im_tensor = cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(im_tensor)

        # 得到输出层数据
        cvOut = self.net.forward()
        # print(cvOut.shape)
        # 得到各个候选框并画出
        for detect in cvOut[0, 0, :, :]:
            score = detect[2]
            if score > 0.5:
                if args.graphic_display == True:
                    left = detect[3] * w
                    top = detect[4] * h
                    right = detect[5] * w
                    bottom = detect[6] * h
                    cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
                    goal = "{:.2f}%".format(score * 100)
                    cv.putText(image, goal, (int(left), int(bottom)), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (23, 230, 210), 2)
                people_num += 1

        # 将人数进行增强
        # people_num*=3

        # 计算出当前FPS值，并显示
        fps = fps + 1
        sfps = fps / (time.time() - t_start)
        cv.putText(image, "FPS : " + str(int(sfps)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(image, "P : " + str(people_num), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if people_num >= args.people_threshold:
            cv.putText(image, "warn!!!!", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 5,
                       (0, 0, 255), 4)

        # 打印当前人数
        newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        PrintStr = place_name + "\tpeople:" + str(people_num) + "\t" + newtime
        # 使用一行输出结果并刷新
        print('\r', "{}".format(PrintStr), end='')

        return people_num, image
