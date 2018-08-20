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
  `mw_num_engines` int(3) DEFAULT '-1',
  PRIMARY KEY (`mw_id`),
  UNIQUE KEY `mw_index_2017_mw_file_hash_uindex` (`mw_file_hash`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=latin1;
"""

sql2 = """
UPDATE `mw_index_2017_%s` set mw_num_engines = -1 where mw_id > 0;
"""

sql3 = """
DROP table `mw_index_2017_%s`;
"""

sql4 = """
rename table mw_2017_%s to mw_index_2017_%s
"""

sql5 = """
CREATE TABLE `mw_2017_section_%s` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `mw_file_hash` varchar(64) NOT NULL,
  `section_name` varchar(60) NOT NULL DEFAULT '',
  `virtual_size` bigint(20) NOT NULL DEFAULT '0',
  `virtual_address` bigint(20) NOT NULL DEFAULT '0',
  `sizeof_raw_data` bigint(20) NOT NULL DEFAULT '0',
  `pointerto_raw_data` bigint(20) NOT NULL DEFAULT '0',
  `ALIGN_1024BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_128BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_16BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_1BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_2048BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_256BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_2BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_32BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_4096BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_4BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_512BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_64BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_8192BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `ALIGN_8BYTES` tinyint(1) NOT NULL DEFAULT '0',
  `CNT_CODE` tinyint(1) NOT NULL DEFAULT '0',
  `CNT_INITIALIZED_DATA` tinyint(1) NOT NULL DEFAULT '0',
  `CNT_UNINITIALIZED_DATA` tinyint(1) NOT NULL DEFAULT '0',
  `GPREL` tinyint(1) NOT NULL DEFAULT '0',
  `LNK_COMDAT` tinyint(1) NOT NULL DEFAULT '0',
  `LNK_INFO` tinyint(1) NOT NULL DEFAULT '0',
  `LNK_NRELOC_OVFL` tinyint(1) NOT NULL DEFAULT '0',
  `LNK_OTHER` tinyint(1) NOT NULL DEFAULT '0',
  `LNK_REMOVE` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_16BIT` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_DISCARDABLE` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_EXECUTE` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_LOCKED` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_NOT_CACHED` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_NOT_PAGED` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_PRELOAD` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_READ` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_SHARED` tinyint(1) NOT NULL DEFAULT '0',
  `MEM_WRITE` tinyint(1) NOT NULL DEFAULT '0',
  `TYPE_NO_PAD` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `mw_2017_section_uindex` (`mw_file_hash`,`section_name`,`pointerto_raw_data`,`virtual_address`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
"""

sql6 = """
rename table mw_2017_section_%s to mw_index_2017_section_%s
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
    execute_sql(sql2, suffix)
    close()
close()
