import os
import lief
import sys

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


def get_sql(table, item):
    """

    :param table:
    :param item:
    :return:
    """
    sql = """INSERT INTO {table} ({K}) VALUES ('{V}') ON DUPLICATE KEY UPDATE """
    keys = ",".join(item.keys())
    values = "','".join(item.values())
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

path = "/ssd/2017/" + sys.argv[1]
directory = os.fsencode(path)

characteristics = {'ALIGN_1024BYTES': 1, 'ALIGN_128BYTES': 1, 'ALIGN_16BYTES': 1, 'ALIGN_1BYTES': 1,
                   'ALIGN_2048BYTES': 1, 'ALIGN_256BYTES': 1, 'ALIGN_2BYTES': 1, 'ALIGN_32BYTES': 1,
                   'ALIGN_4096BYTES': 1, 'ALIGN_4BYTES': 1, 'ALIGN_512BYTES': 1, 'ALIGN_64BYTES': 1,
                   'ALIGN_8192BYTES': 1, 'ALIGN_8BYTES': 1, 'CNT_CODE': 1, 'CNT_INITIALIZED_DATA': 1,
                   'CNT_UNINITIALIZED_DATA': 1, 'GPREL': 1, 'LNK_COMDAT': 1, 'LNK_INFO': 1, 'LNK_NRELOC_OVFL': 1,
                   'LNK_OTHER': 1, 'LNK_REMOVE': 1, 'MEM_16BIT': 1, 'MEM_DISCARDABLE': 1, 'MEM_EXECUTE': 1,
                   'MEM_LOCKED': 1, 'MEM_NOT_CACHED': 1, 'MEM_NOT_PAGED': 1, 'MEM_PRELOAD': 1, 'MEM_READ': 1,
                   'MEM_SHARED': 1, 'MEM_WRITE': 1, 'TYPE_NO_PAD': 1}

get_connection(db)
cursor = _connection.cursor()
cursor.execute("SET NAMES utf8mb4")
idx = 0
sql = ""

for file in os.listdir(directory):
    filename = os.fsdecode(directory.decode("utf-8") + "/" + file.decode("utf-8"))
    mw_file_hash = file.decode("utf-8")[0:64].upper()

    try:
        pe = lief.PE.parse(filename)
        for section in pe.sections:
            item = {
                "mw_file_hash": mw_file_hash,
                "section_name": str(section.name).replace("\\", "\\\\").replace("'", "\\'"),
                "virtual_size": str(section.virtual_size),
                "virtual_address": str(section.virtual_address),
                "sizeof_raw_data": str(section.sizeof_raw_data),
                "pointerto_raw_data": str(section.pointerto_raw_data),
            }

            table = "mw_2017_section_%s" % mw_file_hash[0]

            for chars in section.characteristics_lists:
                key = str(chars).split('.')[-1]
                if key in characteristics:
                    item[key] = '1'

            sql = get_sql(table, item)
            cursor.execute(sql)

        idx += 1

        if idx % 5000 == 0:
            print(idx)
            close()
            get_connection(db)
            cursor = _connection.cursor()
            cursor.execute("SET NAMES utf8mb4")

    except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
        continue

close()
