import os
import sys

import pymysql
from warnings import filterwarnings

_connection = None


def get_connection(db_config):
    """
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
    :return:
    """
    global _connection
    if _connection is not None:
        _connection.close()
    _connection = None


def get_sql(table, item):
    """

    :param table:
    :param item:
    :return:
    """
    sql = """INSERT INTO {table} ({K}) VALUES ("{V}") ON DUPLICATE KEY UPDATE """
    keys = ",".join(item.keys())
    values = "\",\"".join(item.values())
    for key in item.keys():
        sql += "%s=VALUES(%s)," % (key, key)
    sql = sql[:len(sql) - 1].format(table=table, K=keys, V=values)
    return sql


db = {
    'host': '172.26.187.242',
    'username': 'malware',
    'password': 'AcZUBimQXcVlFb58',
    'db': 'malware'
}

path = "/media/malware/" + sys.argv[1]
directory = os.fsencode(path)

get_connection(db)
cursor = _connection.cursor()
cursor.execute("SET NAMES utf8mb4")
idx = 0
for file in os.listdir(directory):
    try:
        item = {}
        sql = ""
        filename = os.fsdecode(path + "/" + file.decode("utf-8"))
        file_arr = file.decode("utf-8")

        item["mw_file_hash"] = file_arr[:64]
        item["mw_file_prefix"] = sys.argv[1]
        item["mw_file_suffix"] = file_arr[64:]
        table = "mw_2017_%s" % file_arr[0]

        if table != "mw_2017_0":
            continue

        sql = get_sql(table, item)
        cursor.execute(sql)

        idx += 1

        if idx % 50000 == 0:
            print(idx)
            _connection.commit()
            close()
            get_connection(db)
            cursor = _connection.cursor()
            cursor.execute("SET NAMES utf8mb4")
    except:
        print(sql)
        raise

_connection.commit()
close()
