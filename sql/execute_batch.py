# -*- coding: utf-8 -*-
# file: execute_batch.py
# author: joddiyzhang@gmail.com
# time: 2018/7/14 3:48 PM
# ------------------------------------------------------------------------

import time

import pymysql
from warnings import filterwarnings

_connection = None


def get_connection(db_config):
    """
    获取连接
    :return:
    """
    global _connection
    if _connection is None:
        _connection = pymysql.connect(host=db_config['host'], user=db_config['username'],
                                      password=db_config['password'],
                                      db=db_config['db'], charset="utf8")
        _connection.autocommit(True)
        filterwarnings('ignore', category=pymysql.Warning)

    return _connection


def close():
    """
    关闭 DB 连接
    :return:
    """
    global _connection
    if _connection is not None:
        _connection.close()
    _connection = None


def execute_sql(sql, t_suffix):
    """

    :param sql:
    :param t_suffix:
    :return:
    """
    start_time = time.time()

    global _connection
    if _connection is None:
        raise Exception("please init db connect first")

    cursor = _connection.cursor()
    cursor.execute("SET NAMES utf8mb4")

    cursor.execute(sql % t_suffix)

    cursor.close()
    print("--- %s seconds ---" % (time.time() - start_time))


sql1 = """
CREATE TABLE `mw_2017_%s` (
  `mw_id` int(11) NOT NULL AUTO_INCREMENT,
  `mw_file_hash` varchar(64) DEFAULT NULL,
  `mw_file_prefix` varchar(64) NOT NULL DEFAULT '',
  `mw_file_suffix` varchar(64) NOT NULL DEFAULT '',
  `mw_num_engines` int(3) DEFAULT '0',
  PRIMARY KEY (`mw_id`),
  UNIQUE KEY `mw_index_2017_mw_file_hash_uindex` (`mw_file_hash`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=latin1;
"""

db = {
    'host': '172.26.187.242',
    'username': 'malware',
    'password': 'AcZUBimQXcVlFb58',
    'db': 'malware'
}

close()
table_suffix = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
res = []
for suffix in table_suffix:
    get_connection(db)
    execute_sql(sql1, suffix)
    close()
close()
