"""
将 customer_clusters_simple.csv 导入 MySQL

用法：
1. 确保 MySQL 已安装并运行
2. 创建数据库：CREATE DATABASE customer_profile;
3. 配置下方 MYSQL_CONFIG 中的主机、用户、密码
4. 执行：python import_csv_to_mysql.py
"""

from __future__ import annotations

import os
import pandas as pd

# MySQL 连接配置（也可通过环境变量覆盖）
MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "customer_profile"),
    "charset": "utf8mb4",
}

CSV_PATH = "customer_clusters_simple.csv"
TABLE_NAME = "customer_clusters"


def main():
    try:
        import pymysql
    except ImportError:
        print("请先安装 pymysql：pip install pymysql")
        return

    if not os.path.exists(CSV_PATH):
        print(f"错误：未找到文件 {CSV_PATH}")
        return

    print(f"读取 CSV：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    db_name = MYSQL_CONFIG["database"]
    # 先连接不含 database 的实例以创建库
    config_no_db = {k: v for k, v in MYSQL_CONFIG.items() if k != "database"}
    # 使用上下文管理器确保连接与游标可靠关闭
    with pymysql.connect(**config_no_db) as conn:
        with conn.cursor() as cur:
            # 1) 创建数据库并切换
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cur.execute(f"USE `{db_name}`")

            # 2) 若表已存在则删除重建（可选：改为 TRUNCATE 保留表结构）
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

            # 3) 根据 DataFrame 列生成建表语句（统一用 TEXT 避免类型问题）
            cols = df.columns.tolist()
            col_defs = ", ".join([f"`{c}` TEXT" for c in cols])
            cur.execute(f"CREATE TABLE {TABLE_NAME} ({col_defs})")

            # 4) 批量插入：构造参数列表后使用 executemany，加快导入速度
            placeholders = ", ".join(["%s"] * len(cols))
            sql = f"INSERT INTO {TABLE_NAME} ({', '.join([f'`{c}`' for c in cols])}) VALUES ({placeholders})"
            params = []
            for _, row in df.iterrows():
                vals = [None if pd.isna(v) else str(v) for v in row]
                params.append(vals)
            if params:
                cur.executemany(sql, params)

            conn.commit()
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            cnt = cur.fetchone()[0]
            print(f"导入完成，共 {cnt} 行，表名：{TABLE_NAME}")


if __name__ == "__main__":
    main()
