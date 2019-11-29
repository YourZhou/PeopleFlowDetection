import cv2 as cv
import tensorflow as tf
import os
import time
import numpy as np
import datetime
from networks import multi_column_cnn


# 人流量密度图显示（人数>=100）
class People_Flow_Density:
    def __init__(self):
        self.set_gpu(1)

        self.img_path = ''
        self.model_path = './model/MCNN_model/v1-2050'
        # crop_size = 256

        # place holder位置保持器(定义变量)
        self.input_img_placeholder = tf.placeholder(tf.float32, shape=([None, None, None, 3]))
        self.density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

        self.inference_density_map = multi_column_cnn(self.input_img_placeholder)

    def set_gpu(self, gpu=0):
        """
        the gpu used setting
        :param gpu: gpu id
        :return:
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # 密度图生成
    def image_processing(self, input):
        # 高斯模糊
        kernel_size = (3, 3)
        sigma = 15
        r_img = cv.GaussianBlur(input, kernel_size, sigma)

        # 灰度图标准化
        norm_img = np.zeros(r_img.shape)
        norm_img = cv.normalize(r_img, norm_img, 0, 255, cv.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)
        # r_img = cv.resize(r_img, (720, 420))
        # utils.show_density_map(r_img)

        # 灰度图颜色反转
        imgInfo = norm_img.shape
        heigh = imgInfo[0]
        width = imgInfo[1]
        dst = np.zeros((heigh, width, 1), np.uint8)
        for i in range(0, heigh):
            for j in range(0, width):
                grayPixel = norm_img[i, j]
                dst[i, j] = 255 - grayPixel

        # 生成热力图
        heat_img = cv.applyColorMap(dst, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
        output = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

        return output

    # 密度图与原图叠加
    def image_add_heatmap(self, frame, heatmap, alpha=0.5):
        img_size = frame.shape
        heatmap = cv.resize(heatmap, (img_size[1], img_size[0]))
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
        cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图
        return frame

    def density_infer(self, input_img, args):
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

        ori_crowd_img = input_img
        # h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]
        img = ori_crowd_img.reshape((ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            result = sess.run(self.inference_density_map, feed_dict={self.input_img_placeholder: [(img - 127.5) / 128]})

        people_num = int(result.sum())
        dmap_img = result[0, :, :, 0]

        final_img = self.image_processing(dmap_img)
        final_img = self.image_add_heatmap(ori_crowd_img, final_img, 0.5)

        cv.putText(final_img, "P : " + str(people_num), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 计算出当前FPS值，并显示
        fps = fps + 1
        sfps = fps / (time.time() - t_start)
        cv.putText(final_img, "FPS : " + str(int(sfps)), (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if people_num >= args.people_threshold:
            cv.putText(final_img, "warn!!!!", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 5,
                       (0, 0, 255), 4)

        # 打印当前人数
        newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        PrintStr = place_name + "\tpeople:" + str(people_num) + "\t" + newtime
        # 使用一行输出结果并刷新
        print('\r', "{}".format(PrintStr), end='')
        # cv.imshow("really", final_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return people_num, final_img
