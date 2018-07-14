# -*- coding: utf-8 -*-
# file: section.py
# author: joddiyzhang@gmail.com
# time: 2018/7/5 10:52 AM
# ------------------------------------------------------------------------

# -*- coding: utf-8 -*-
# file: new.py
# author: joddiyzhang@gmail.com
# time: 2018/7/3 11:15 AM
# ------------------------------------------------------------------------
import os
import lief
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


def common_set(table, data, interval=5000):
    """
    通用插入方法，当唯一键重复时更新
    :return:
    """
    global _connection
    if _connection is None:
        raise Exception("please init db connect first")

    cursor = _connection.cursor()
    cursor.execute("SET NAMES utf8mb4")

    # try:

    cnt = 0
    items = []
    is_first = True
    # SQL 插入语句
    sql = ""
    for item in data:
        if item is None:
            break
        if is_first:
            sql = """INSERT INTO {table} ({K}) VALUES ({V}) ON DUPLICATE KEY UPDATE """
            keys = ",".join(item.keys())
            values = "%s," * (len(item) - 1) + "%s"
            for key in item.keys():
                sql += "%s=VALUES(%s)," % (key, key)
            sql = sql[:len(sql) - 1].format(table=table, K=keys, V=values)
            is_first = False
        if cnt >= interval:
            cursor.executemany(sql, items)
            cnt = 0
            items.clear()
        else:
            items.append(list(item.values()))
            cnt += 1
    if cnt > 0:
        cursor.executemany(sql, items)
        # except:
        #     _connection.rollback()

    cursor.close()
    # _connection.close()


db = {
    'host': '172.26.187.242',
    'username': 'malware',
    'password': 'AcZUBimQXcVlFb58',
    'db': 'malware'
}

path = "/hdd1/raw_pe_data/2017/" + sys.argv[0]
directory = os.fsencode(path)

get_connection(db)
cursor = _connection.cursor()
cursor.execute("SET NAMES utf8mb4")

info_set = []
inte = 50000
current_table = None
idx = 0

for file in os.listdir(directory):
    filename = os.fsdecode(directory + "/" + file)
    file_arr = file.split("_")
    mw_file_hash = file_arr[0].upper()
    try:
        pe = lief.PE.parse(filename)
        for section in pe.sections:
            tmp_table = mw_file_hash[0]
            if current_table is None:
                current_table = tmp_table
            else:
                if tmp_table != current_table or len(info_set) >= inte:
                    print(idx)
                    get_connection(db)
                    common_set("mw_index_2017_{0}_section".format(current_table), info_set, interval=inte)
                    close()
                    info_set = []
                    current_table = tmp_table
            item = {
                "mw_id": mw_file_hash,
                "section_name": section.name,
                "virtual_size": section.virtual_size,
                "virtual_address": section.virtual_address,
                "sizeof_raw_data": section.sizeof_raw_data,
                "pointerto_raw_data": section.pointerto_raw_data,
                "entropy": section.entropy,
            }
            for chars in section.characteristics_lists:
                key = str(chars).split('.')[-1]
                item[key] = 1
            info_set.append(item)
            idx += 1

    except (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
        print("lief error: ", str(e))
        continue
    except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
        continue

get_connection(db)
common_set("mw_index_2017_{0}_section".format(current_table), info_set, interval=inte)
close()
