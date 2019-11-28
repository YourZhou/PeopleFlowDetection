# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/25
    @Comment :
"""

"""
导入必要库文件
"""
import cv2 as cv
import tensorflow as tf
import pymysql
import os
import datetime
import time
import numpy as np
import threading as td
import multiprocessing as mp
from networks import multi_column_cnn
import requests
import argparse
import functools
from queue import Queue
from utility import add_arguments, print_arguments

"""
获得命令行参数
graphic_display：是否打开图形界面
use_video：是否使用视频检测（否则调用摄像头）
video_path：当使用视频时，视频地址
"""
np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('graphic_display', bool, True, "Whether use display.")
add_arg('use_video', bool, True, "Whether use video.")
add_arg('video_path', str, './all_test01.mp4', "The video used to inference and visualize.")


# yapf: enable

# 连接智慧景区数据库及相关操作
class Scenic_mysql_conn:
    """
    数据传输操作
    ip：47.102.153.115/3306
    user='root',
    passwd='1234',
    db='scenic_area',
    charset='utf8

    place_name：地点名称
    people_num：人数
    sql_id：数据库id
    """

    def __init__(self):
        self.place_name = "tower"
        self.people_num = 0
        self.id = 1
        self.place_num = 1
        self.sql_id = "tbl_tower"

    def getConnecttion(self):
        connection = pymysql.Connect(host='47.102.153.115',
                                     port=3306,
                                     user='root',
                                     passwd='1234',
                                     db='scenic_area',
                                     charset='utf8')
        return connection

    def set_place_name(self, place_num):
        """
        :param place_num: 输入地点id，初始化数据库配置信息
        :return:
        """
        Scenic_mysql_conn.place_num = place_num
        if Scenic_mysql_conn.place_num == '1':
            Scenic_mysql_conn.sql_id = "tbl_tower"
            Scenic_mysql_conn.place_name = "tower"
        elif Scenic_mysql_conn.place_num == '2':
            Scenic_mysql_conn.sql_id = "tbl_rock"
            Scenic_mysql_conn.place_name = "rock"
        elif Scenic_mysql_conn.place_num == '3':
            Scenic_mysql_conn.sql_id = "tbl_ruins"
            Scenic_mysql_conn.place_name = "ruins"
        else:
            print("Worry")
        print(Scenic_mysql_conn.place_num + '\t' + Scenic_mysql_conn.place_name + '\t' + Scenic_mysql_conn.sql_id)

    def set_people_num(self, num):
        Scenic_mysql_conn.people_num = num

    def get_place_name(self):
        return Scenic_mysql_conn.place_name

    def get_people_num(self):
        return Scenic_mysql_conn.people_num

    def get_sql_id(self):
        return Scenic_mysql_conn.sql_id

    def set_id(self, id):
        Scenic_mysql_conn.id = id

    def add_id(self):
        Scenic_mysql_conn.id += 1

    def get_id(self):
        return Scenic_mysql_conn.id

    def get_place_num(self):
        return Scenic_mysql_conn.place_num

    def conn_to_sql(self, place_num):
        """
        连接数据库操作
        :param place_num:传入要连接的数据库
        :return:
        """
        Scenic_mysql_conn.set_place_name(self, place_num)
        self.conn = Scenic_mysql_conn.getConnecttion(self)
        self.cursor = self.conn.cursor()

    def setting_to_sql(self):
        """
        上传实时人数数量
        :return:
        """
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("insert into %s(last_date,people,place_name) value('%s','%d','%s')" %
                            (Scenic_mysql_conn.get_sql_id(self), dt,
                             Scenic_mysql_conn.get_people_num(self),
                             Scenic_mysql_conn.get_place_name(self)))
        self.conn.commit()

    def close_to_sql(self):
        """
        关闭数据库连接
        :return:
        """
        self.conn.close()
        self.cursor.close()

    def update_to_sql(self):
        """
        更新数据库人数信息
        :return:
        """
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("update tbl_place set last_people='%d' where place_id=%s" %
                            (Scenic_mysql_conn.get_people_num(self),
                             Scenic_mysql_conn.get_place_num(self)))
        self.cursor.execute("update tbl_place set last_date='%s' where place_id=%s" %
                            (dt, Scenic_mysql_conn.get_place_num(self)))
        self.conn.commit()

    def select_to_people_threshold(self):
        """
        查看数据库得到云端设置的人数阈值
        :return:人数阈值
        """
        sql = "select config_value from tbl_config where config_name='阈值'"
        self.cursor.execute(sql)
        rs = self.cursor.fetchone()
        for row in rs:
            row = int(row)
        return row


class People_Flow_Density:
    def __init__(self):
        self.set_gpu(1)

        self.img_path = 'D:\\YourZhouProject\\mcnn_project\\pytorch_mcnn\\part_A_final\\test_data\\images\\IMG_67.jpg'
        self.model_path = 'D:\\YourZhouProject\\mcnn_project\\tf_mcnn\\MCNN_REPRODUCTION-master\\ckpts2\\mcnn\\v1-1425'
        # crop_size = 256

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

    def density_infer(self, input_img):
        ori_crowd_img = cv.imread(self.img_path)
        # h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]
        img = ori_crowd_img.reshape((ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))

        # place holder位置保持器
        input_img_placeholder = tf.placeholder(tf.float32, shape=([None, None, None, 3]))
        density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

        inference_density_map = multi_column_cnn(input_img_placeholder)

        saver = tf.train.Saver()

        time_star = time.time()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            result = sess.run(inference_density_map, feed_dict={input_img_placeholder: [(img - 127.5) / 128]})

        time_over = time.time() - time_star
        print(time_over)

        num = result.sum()
        print(num)
        dmap_img = result[0, :, :, 0]

        final_img = self.image_processing(dmap_img)
        final_img = self.image_add_heatmap(ori_crowd_img, final_img, 0.5)

        cv.putText(final_img, "P : " + str(int(num)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("really", final_img)

        cv.waitKey(0)
        cv.destroyAllWindows()


# 录制警报视频以及视频的上传
def save_and_send_video(mp_q, place_names, people_nums, graphic_display):
    """
    当出现人数预警，进行预警视频录制
    并上传服务器
    :param mp_q: 传入多进程的管道
    :param place_names: 地点信息
    :param people_nums: 人数信息
    :param graphic_display: 是否打开图形界面
    :return:
    """
    # 服务器接收预警视频接口
    upload_url = 'http://47.102.153.115:8080/isa/warn'
    # 视频时间
    Video_time = 5

    # 得到当前时间
    newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("-------waiting------\n"
          "   预警视频录制中\n"
          "--------------------\n")

    # # 创建线程进行计时
    # video_time = td.Thread(target=recording_time, daemon=True)
    # # 计时开始（线程开启）
    # video_time.start()

    # 初始化录制信息（文件夹建立）
    video_path = "./warn_video/"
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    video_name = video_path + newtime + ".mp4"

    # 文件编码
    fourcc = cv.VideoWriter_fourcc(*"AVC1")
    # 以720x420录制（每秒10帧）
    out = cv.VideoWriter(video_name, fourcc, 10.0, (720, 420))

    # 获得录制开始时间
    Recording_time = 0
    t_start = time.time()
    # 开始录制
    while Recording_time <= Video_time:
        # 读取视频播放管道数据
        if mp_q.empty() == False:
            image = mp_q.get()
            image = cv.flip(image, 1)
            # gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            new_img = cv.resize(image, (720, 420))
            out.write(new_img)
            # 如打开图形化界面则显示
            if graphic_display == True:
                cv.imshow("video", new_img)
                cv.waitKey(10)
            Recording_time = time.time() - t_start
    # 关闭录制
    out.release()
    if graphic_display == True:
        cv.destroyAllWindows()

    print("-------waiting------\n"
          "   预警视频发送中..\n"
          "--------------------\n")

    # 文件发送
    files = {'file': open(video_name, 'rb')}
    upload_data = {"warn_place": place_names, "people": people_nums, "fileName": video_name, "uoType": 1}
    upload_res = requests.post(upload_url, data=upload_data, files=files)
    # print(upload_res.text)

    print("----------------------------\n"
          "     预警视频发送成功！！\n"
          "----------------------------\n")


# 视频后台播放
def image_put(q, args):
    """
    加载视频流文件
    :param q: 多进程读取视频的管道
    :param args: 传入初始化参数（是否使用视频）
    :return:
    """
    # 判断是否使用视频
    if args.use_video == False:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(args.video_path)

    # 输出查看
    if cap.isOpened():
        print('\nvideo ok')
    else:
        print('\nvideo error')

    # 后台进程读取视频
    while True:
        q.put(cap.read()[1])
        # 不断刷新缓冲区
        q.get() if q.qsize() > 1 else time.sleep(0.01)


# 数据上传数据库
def sql_wait_time(q, td_threshold, place_num):
    """
    读取数据库阈值信息
    并每隔3秒上传云服务器
    本函数为线程函数
    :param q:加载人数信息的管道
    :param td_threshold:阈值信息存放的管道
    :param place_num:地点位置信息
    :return:
    """
    # 存放人数的缓冲区
    num_buff = 0

    #### MySQL连接 #####
    msql_conn = Scenic_mysql_conn()
    msql_conn.conn_to_sql(place_num)

    # 获取当前初始阈值信息,并存入管道
    people_threshold = msql_conn.select_to_people_threshold()
    td_threshold.put(people_threshold)

    # 每3秒上报一次人数入数据库
    while True:
        time.sleep(3)
        num_buff = q.get()
        # 当收到num_buff=1000的信号关闭连接
        if num_buff == 1000:
            msql_conn.close_to_sql()
            break
        people_num = num_buff
        # 进行人数上报
        msql_conn.set_people_num(people_num)
        msql_conn.setting_to_sql()
        msql_conn.update_to_sql()
        # 查询阈值是否变化
        lot_buf = int(msql_conn.select_to_people_threshold())
        if people_threshold != lot_buf:
            people_threshold = lot_buf
            td_threshold.put(people_threshold)


# 视频录制定时（延时5s）
def recording_time():
    """
    定时5秒钟做视频录制
    :return:
    """
    time.sleep(5)


# 多进程设置
def td_mp_set(args, place_num):
    """
    进程、线程管理
    :param args: 参数
    :param place_num:地点位置
    :return: 三个管道
    """
    # 线程管道x2
    td_queue = Queue()
    td_threshold = Queue()

    # 进程管道x1
    mp_queue = mp.Queue()

    # 数据库传输线程
    wait_send = td.Thread(target=sql_wait_time, args=(td_queue, td_threshold, place_num), daemon=True)
    # 视频读取进程
    video_input = mp.Process(target=image_put, args=(mp_queue, args), daemon=True)
    # 启动
    video_input.start()
    wait_send.start()
    return td_queue, mp_queue, td_threshold


# 总部
def headquarters(td_q, mp_q, td_threshold, place_num, args):
    """
    主要执行程序
    :param td_q: TQ1
    :param mp_q: MQ1
    :param td_threshold: TQ2
    :param place_num: 地点位置
    :param args: 参数传入
    :return:
    """
    # 获取当前时间，实现FPS的显示
    t_start = time.time()
    fps = 0
    if place_num == '1':
        place_name = "tower"
    elif place_num == '2':
        place_name = "rock"
    elif place_num == '3':
        place_name = "ruins"
    else:
        place_name = "rock"
    # 导入及读取tf模型
    inference_pb = "./frozen_inference_graph.pb"
    graph_txt = "./graph.pbtxt"
    net = cv.dnn.readNetFromTensorflow(inference_pb, graph_txt)
    # 从queue获取当前阈值信息
    while True:
        if td_threshold.empty() == False:
            people_threshold = td_threshold.get()
            print("*****************************\n"
                  "people_threshold=%s"
                  "\n*****************************" % people_threshold)
            break
    # 当前视频模式，1：正常检测模式；2：视频录制发送模式
    Video_Pattern = 1
    # 数据连接状态
    Sql_state = 1
    # 视频时间记号（3s/n）
    lot_time = 0
    # 当前视频内人数
    people_num = 0
    # 是否为第一次视频
    first_video = True
    # 时间到达，数据上传标志
    wait = "0"
    graphic_display = args.graphic_display

    while True:
        # 正常检测模式
        while Video_Pattern == 1:
            # 是否到达人数上报时间
            if td_threshold.empty() == False:
                people_threshold = td_threshold.get()
                print("*****************************\n"
                      "people_threshold change to %d"
                      "\n*****************************" % people_threshold)
            # 人数初始化
            people_num = 0
            # 从进程中获取最新帧
            image = mp_q.get()
            image = cv.flip(image, 1)
            # 获取视频宽高
            h, w = image.shape[:2]
            # 将图片导入模型进行处理
            im_tensor = cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
            net.setInput(im_tensor)
            # 得到输出层数据
            cvOut = net.forward()
            # print(cvOut.shape)
            # 得到各个候选框并画出
            for detect in cvOut[0, 0, :, :]:
                score = detect[2]
                if score > 0.5:
                    if graphic_display == True:
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

            if people_num >= people_threshold:
                cv.putText(image, "warn!!!!", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 5,
                           (0, 0, 255), 4)

            # 打印当前人数
            newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            PrintStr = place_name + "\tpeople:" + str(people_num) + "\t" + newtime
            # 使用一行输出结果并刷新
            print('\r', "{}".format(PrintStr), end='')

            # 进行时间标记
            if td_q.empty() == True:
                lot_time += 1
            td_q.put(people_num)
            # 不断冲刷缓冲区，保持最新人数
            if td_q.qsize() > 1:
                td_q.get()

            # 进行预警人数上报的判定
            if ((first_video == True or lot_time > 10) and people_num >= people_threshold):
                lot_time = 0
                Video_Pattern = 2
                first_video = False

            if graphic_display == True:
                # 将视频进行输出
                # 将视频尺寸进行修剪
                new_img = cv.resize(image, (720, 420))
                cv.imshow("detection-out", new_img)

            if (cv.waitKey(10) == 27):
                Sql_state = 0
                break

        while Video_Pattern == 2:
            save_and_send_video(mp_q, place_name, people_num, graphic_display)
            Video_Pattern = 1
            if (cv.waitKey(10) == 27):
                Sql_state = 0
                break

        if Sql_state == 0:
            td_q.put(1000)
            break


def main():
    args = parser.parse_args()
    print_arguments(args)

    # place_name = "null"
    # sql_id = '0'

    print("1:普贤塔  2：象山岩  3：桂林抗战遗址")
    place_num = input("请输入地点标号：")

    # if place_num == '1':
    #     sql_id = "tbl_tower"
    #     place_name = "tower"
    # elif place_num == '2':
    #     sql_id = "tbl_rock"
    #     place_name = "rock"
    # elif place_num == '3':
    #     sql_id = "tbl_ruins"
    #     place_name = "ruins"
    # else:
    #     print("Worry")

    td_q, mp_q, td_threshold = td_mp_set(args, place_num)
    headquarters(td_q, mp_q, td_threshold, place_num, args)


if __name__ == '__main__':
    main()
