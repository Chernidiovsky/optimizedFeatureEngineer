import pymysql
import params as config
import pandas as pd
from sqlalchemy import create_engine
import traceback


def getCursor(db):
    connection = pymysql.connect(host=config.mysql_host,
                                 port=3306,
                                 user=config.mysql_user,
                                 password=config.mysql_password,
                                 db=db,
                                 charset='utf8')
    return connection, connection.cursor()


def getDfFromSql(db, sql):
    engine = create_engine("mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8" % (
        config.mysql_user, config.mysql_password, config.mysql_host, db))
    return pd.read_sql(sql, engine)


def executeSql(db, sql):
    con, cursor = getCursor(db)
    try:
        cursor.execute(sql)
        con.commit()
    except:
        con.rollback()
        traceback.print_exc()


def saveDf(df, db, tableName, isAppend=True, index=None):
    engine = create_engine("mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8" % (
        config.mysql_user, config.mysql_password, config.mysql_host, db))
    assert isinstance(df, pd.DataFrame)
    saveIndex = True
    if index is None:
        saveIndex = False
    else:
        df = df.set_index(index)
    mode = 'append' if isAppend else 'replace'
    df.to_sql(tableName, engine, if_exists=mode, index=saveIndex, index_label=index)
