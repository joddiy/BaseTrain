# -*- coding: utf-8 -*-
# file: new.py
# author: joddiyzhang@gmail.com
# time: 2018/7/3 11:15 AM
# ------------------------------------------------------------------------
import os
import sys

import pymysql
from warnings import filterwarnings

_connection = None


# 此类只能存储一个连接，之后考虑升级

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


def get_update_sql(table, mw_file_size, mw_file_directory, mw_file_hash):
    sql = """ UPDATE mw_index_2017_{table} SET mw_file_sze = {A}, mw_file_directory = '{B}', mw_sections_num = {C} WHERE mw_file_hash = '{C}'"""
    sql = sql.format(table=table, A=mw_file_size, B=mw_file_directory, C=mw_file_hash)
    return sql


db = {
    'host': '172.26.187.242',
    'username': 'malware',
    'password': 'AcZUBimQXcVlFb58',
    'db': 'malware'
}

path = "/hdd1/raw_pe_data/2017/" + sys.argv[1]
directory = os.fsencode(path)

get_connection(db)
cursor = _connection.cursor()
cursor.execute("SET NAMES utf8mb4")

sql_list = []
idx = 0
for file in os.listdir(directory):
    filename = os.fsdecode(path + "/" + file.decode("utf-8"))
    file_arr = file.decode("utf-8").split("_")
    mw_file_directory = path
    mw_file_hash = file_arr[0].upper()
    mw_file_size = file_arr[1]
    table = mw_file_hash[0]
    sql = get_update_sql(table, mw_file_size, mw_file_directory.decode("utf-8"), mw_file_hash)
    cursor.execute(sql)
    if idx % 1000 == 0:
        print(idx)
        _connection.commit()
        close()
        get_connection(db)
        cursor = _connection.cursor()
        cursor.execute("SET NAMES utf8mb4")
    idx += 1

_connection.commit()
close()
