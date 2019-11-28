import pymysql
import datetime


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
