# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/25
    @Comment :
"""

"""
导入必要库文件
"""
import threading as td
import multiprocessing as mp
import requests
import argparse
import functools
from SSDdetection import *
from MCNNdensity import *
from MySQLconn import *
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
add_arg('video_path', str, './test_video/all_video.mp4', "The video used to inference and visualize.")
add_arg('monitoring_place', str, '1', "1:普贤塔(tower)  2：象山岩(rock)  3：桂林抗战遗址(ruins)")
add_arg('people_threshold', int, 30, "Set the threshold number of people to alert.")


# yapf: enable


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
def sql_wait_time(q, td_threshold, args):
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
    msql_conn.conn_to_sql(args.monitoring_place)

    # 获取当前初始阈值信息,并存入管道
    args.people_threshold = msql_conn.select_to_people_threshold()
    td_threshold.put(args.people_threshold)

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
        if args.people_threshold != lot_buf:
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
def td_mp_set(args):
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
    wait_send = td.Thread(target=sql_wait_time, args=(td_queue, td_threshold, args), daemon=True)
    # 视频读取进程
    video_input = mp.Process(target=image_put, args=(mp_queue, args), daemon=True)
    # 启动
    video_input.start()
    wait_send.start()
    return td_queue, mp_queue, td_threshold


# 总部
def headquarters(td_q, mp_q, td_threshold, args):
    """
    主要执行程序
    :param td_q: TQ1
    :param mp_q: MQ1
    :param td_threshold: TQ2
    :param args: 参数传入
    :return:
    """
    place_num = args.monitoring_place
    if place_num == '1':
        place_name = "tower"
    elif place_num == '2':
        place_name = "rock"
    elif place_num == '3':
        place_name = "ruins"
    else:
        place_name = "rock"

   # mPeopleDetection = People_detection()
    mPeopleDensity = People_Flow_Density()

    # 从queue获取当前阈值信息
    while True:
        if td_threshold.empty() == False:
            args.people_threshold = td_threshold.get()
            print("*****************************\n"
                  "people_threshold=%s"
                  "\n*****************************" % args.people_threshold)
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
                args.people_threshold = td_threshold.get()
                print("*****************************\n"
                      "people_threshold change to %d"
                      "\n*****************************" % args.people_threshold)

            # 从进程中获取最新帧
            image = mp_q.get()

            # people_num, image = mPeopleDetection.detection(image, args)
            people_num, image = mPeopleDensity.density_infer(image, args)

            # 进行时间标记
            if td_q.empty() == True:
                lot_time += 1
            td_q.put(people_num)
            # 不断冲刷缓冲区，保持最新人数
            if td_q.qsize() > 1:
                td_q.get()

            # 进行预警人数上报的判定
            if ((first_video == True or lot_time > 10) and people_num >= args.people_threshold):
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
    args.monitoring_place = place_num
    # print_arguments(args)

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

    td_q, mp_q, td_threshold = td_mp_set(args)
    headquarters(td_q, mp_q, td_threshold, args)


if __name__ == '__main__':
    main()
