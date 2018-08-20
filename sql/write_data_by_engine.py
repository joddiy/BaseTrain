import pymysql
from warnings import filterwarnings
import pandas as pd

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


def get_sql(table, mw_num_engines, mw_file_hash):
    sql = """ UPDATE {table} SET mw_num_engines='{A}' WHERE mw_file_hash='{B}'"""
    sql = sql.format(table=table, A=mw_num_engines, B=mw_file_hash)
    return sql


db = {
    'host': '172.26.187.242',
    'username': 'malware',
    'password': 'AcZUBimQXcVlFb58',
    'db': 'malware'
}

train_y = pd.read_csv("/ssd/2017/new_db201701-201801.txt", sep="\t", error_bad_lines=False)
print(train_y.shape)

get_connection(db)
cursor = _connection.cursor()
cursor.execute("SET NAMES utf8mb4")
for idx, row in train_y.iterrows():
    try:
        sql = ""

        table = "mw_index_2017_%s" % row['FileHash'][0].upper()

        sql = get_sql(table, str(row['numEngines']), str(row['FileHash']))
        cursor.execute(sql)

        if idx % 5000 == 0:
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