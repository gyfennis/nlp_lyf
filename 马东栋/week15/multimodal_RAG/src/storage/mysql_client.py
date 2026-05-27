from mysql.connector import pooling
from src.config import get_mysql_config


class MySQLClient:
    def __init__(self, host=None, port=None, username=None, password=None, database=None, pool_size=None):
        cfg = get_mysql_config()
        self.config = {
            "host": host or cfg["host"],
            "port": port or cfg["port"],
            "user": username or cfg["username"],
            "password": password or cfg["password"],
            "database": database or cfg["database"],
            "pool_name": "myrag_pool",
            "pool_size": pool_size or cfg.get("pool_size", 10),
            "pool_reset_session": True,
        }
        self.pool = pooling.MySQLConnectionPool(**self.config)

    def get_connection(self):
        return self.pool.get_connection()

    def execute(self, query: str, params: tuple = None):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def execute_many(self, query: str, params_list: list):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.executemany(query, params_list)
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    def execute_insert(self, query: str, params: tuple = None):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
        finally:
            cursor.close()
            conn.close()
