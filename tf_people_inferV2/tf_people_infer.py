import cv2 as cv
import pymysql
import datetime
import time
import threading as td
import multiprocessing as mp
import requests
import argparse
import functools
from queue import Queue
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('graphic_display', bool, True, "Whether use display.")
add_arg('use_video', bool, True, "Whether use video.")
add_arg('video_path', str, './all_test01.mp4', "The video used to inference and visualize.")


# yapf: enable

# 连接智慧景区数据库及相关操作
class Scenic_mysql_conn:
    place_name = "tower"
    people_num = 0
    id = 1
    place_num = 1
    sql_id = "tbl_tower"

    def getConnecttion(self):
        connection = pymysql.Connect(host='47.102.153.115',
                                     port=3306,
                                     user='root',
                                     passwd='1234',
                                     db='scenic_area',
                                     charset='utf8')
        return connection

    def set_place_name(self, place_num):
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
        Scenic_mysql_conn.set_place_name(self, place_num)
        self.conn = Scenic_mysql_conn.getConnecttion(self)
        self.cursor = self.conn.cursor()

    def setting_to_sql(self):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("insert into %s(last_date,people,place_name) value('%s','%d','%s')" %
                            (Scenic_mysql_conn.get_sql_id(self), dt,
                             Scenic_mysql_conn.get_people_num(self),
                             Scenic_mysql_conn.get_place_name(self)))
        self.conn.commit()

    def close_to_sql(self):
        self.conn.close()
        self.cursor.close()
    def update_to_sql(self):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("update tbl_place set last_people='%d' where place_id=%s" %
                            (Scenic_mysql_conn.get_people_num(self),
                             Scenic_mysql_conn.get_place_num(self)))
        self.cursor.execute("update tbl_place set last_date='%s' where place_id=%s" %
                            (dt, Scenic_mysql_conn.get_place_num(self)))
        self.conn.commit()

    def select_to_people_threshold(self):
        sql = "select config_value from tbl_config where config_name='阈值'"
        self.cursor.execute(sql)
        rs = self.cursor.fetchone()
        for row in rs:
            row = int(row)
        return row


# 录制警报视频以及视频的上传
def save_and_send_video(mp_q, place_names, people_nums, graphic_display):
    newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("-------waiting------\n"
          "   预警视频录制中\n"
          "--------------------\n")
    video_time = td.Thread(target=recording_time, daemon=True)
    video_time.start()
    video_name = "./warn_video/" + newtime + ".mp4"
    fourcc = cv.VideoWriter_fourcc(*"AVC1")
    out = cv.VideoWriter(video_name, fourcc, 10.0, (720, 420))
    while video_time.isAlive() == True:
        if mp_q.empty() == False:
            people_num_v = 0
            image = mp_q.get()
            image = cv.flip(image, 1)
            # gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            new_img = cv.resize(image, (720, 420))
            out.write(new_img)
            if graphic_display == True:
                cv.imshow("video", new_img)
                cv.waitKey(10)
    out.release()
    if graphic_display == True:
        cv.destroyAllWindows()
    print("-------waiting------\n"
          "   预警视频发送中..\n"
          "--------------------\n")
    upload_url = 'http://47.102.153.115:8080/isa/warn'
    files = {'file': open(video_name, 'rb')}
    upload_data = {"warn_place": place_names, "people": people_nums, "fileName": video_name, "uoType": 1}
    upload_res = requests.post(upload_url, data=upload_data, files=files)
    # print(upload_res.text)
    print("----------------------------\n"
          "     预警视频发送成功！！\n"
          "----------------------------\n")


def image_put(q, args):
    if args.use_video == False:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(args.video_path)

    if cap.isOpened():
        print('\nvideo ok')
    else:
        print('\nvideo error')
    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def sql_wait_time(q, td_threshold, place_num):
    num_buff = 0
    #### MySQL连接 #####
    msql_conn = Scenic_mysql_conn()
    msql_conn.conn_to_sql(place_num)
    # 获取当前阈值信息,并存入queue
    people_threshold = msql_conn.select_to_people_threshold()
    td_threshold.put(people_threshold)
    while True:
        time.sleep(3)
        num_buff = q.get()
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


def recording_time():
    time.sleep(5)


def td_mp_set(args, place_num):
    td_queue = Queue()
    mp_queue = mp.Queue()
    td_threshold = Queue()
    wait_send = td.Thread(target=sql_wait_time, args=(td_queue, td_threshold, place_num), daemon=True)
    video_input = mp.Process(target=image_put, args=(mp_queue, args), daemon=True)
    video_input.start()
    wait_send.start()
    return td_queue, mp_queue, td_threshold


def main(td_q, mp_q, td_threshold, place_num, args):
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

            if people_num >= people_threshold:
                cv.putText(image, "warn!!!!", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 5,
                           (0, 0, 255), 4)

            # 打印当前人数
            newtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            print(place_name + "\tpeople:" + str(people_num) + "\t" + newtime)
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

            # 计算出当前FPS值，并显示
            fps = fps + 1
            sfps = fps / (time.time() - t_start)
            cv.putText(image, "FPS : " + str(int(sfps)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    place_name = "null"
    sql_id = '0'
    print("1:普贤塔  2：象山岩  3：桂林抗战遗址")
    place_num = input("请输入地点标号：")
    if place_num == '1':
        sql_id = "tbl_tower"
        place_name = "tower"
    elif place_num == '2':
        sql_id = "tbl_rock"
        place_name = "rock"
    elif place_num == '3':
        sql_id = "tbl_ruins"
        place_name = "ruins"
    else:
        print("Worry")

    td_q, mp_q, td_threshold = td_mp_set(args, place_num)
    main(td_q, mp_q, td_threshold, place_num, args)
