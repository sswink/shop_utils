import json
import sqlite3
import time
import threading
import functools
def synchronized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return wrapper
class Authorize_Tb:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dbname = 'RH.db'



    #初始化
    def init_tb(self):
        conn = sqlite3.connect(self.dbname)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS rh_record
            (id INTEGER PRIMARY KEY     autoincrement,
            work_flow_id        varchar(100),
            call_back_id           varchar(100),
            request_json           TEXT,
            status              varchar(20),
            createtime         Int);
        ''')
        conn.commit()
        conn.close()

    #添加
    @synchronized
    def saveRecord(self,work_flow_id,call_back_id,request_json,status):
        self.init_tb()
        json_string = json.dumps(request_json, ensure_ascii=False)
        conn = sqlite3.connect(self.dbname)
        cur = conn.cursor()
        cur.execute("insert into rh_record (work_flow_id,call_back_id,request_json,createtime,status) values (?,?,?,?,?)",(work_flow_id,call_back_id,json_string,int(time.time()),status))

        conn.commit()
        conn.close()
        return cur.lastrowid

        #查询
    @synchronized
    def find_latest_record_by_work_flow_id(self,work_flow_id):
        self.init_tb()
        conn = sqlite3.connect(self.dbname)
        cur = conn.cursor()
        cur.execute("select id,work_flow_id,call_back_id,expirestime,createtime,status from rh_record where work_flow_id = ? order by createtime desc limit 1",(work_flow_id,))
        info = cur.fetchone()
        conn.close()
        return info

    # 更新
    @synchronized
    def update_status_by_id(self, id, status):
        self.init_tb()
        conn = sqlite3.connect(self.dbname)
        cur = conn.cursor()
        cur.execute("UPDATE rh_record SET status = ? WHERE id = ?",
                    (status, id))
        conn.commit()
        conn.close()