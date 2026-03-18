"""
将 `customer_features_rfmbc.csv` 导入 MySQL（强类型表结构 + 索引）

用法：
1. 确保 MySQL 已安装并运行
2. 创建数据库：CREATE DATABASE customer_profile;
3. 配置下方 MYSQL_CONFIG（或使用环境变量覆盖）
4. 执行导入：python import_csv_to_mysql.py

可选环境变量：
- MYSQL_TABLE：表名（默认 customer_profiles_rfmbc）
- RESET_TABLE：是否先 TRUNCATE（默认 1）
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

CSV_PATH = "customer_features_rfmbc.csv"
# 默认表名（也可通过环境变量覆盖）
TABLE_NAME = os.environ.get("MYSQL_TABLE", "customer_profiles_rfmbc")
# 是否在导入前清空表（TRUNCATE），默认开启
RESET_TABLE = os.environ.get("RESET_TABLE", "1").lower() in ("1", "true", "yes")


_INT_COLS = {
    "Total_Orders",
    "Customer_Lifetime_Days",
    "Days_Since_Last_Purchase",
    "Unique_Products_Purchased",
    "Total_Likes",
    "Total_Shares",
    "Total_Add_to_Cart",
    "Cluster",
}

_FLOAT_COLS = {
    "Total_Sales",
    "Avg_Order_Value",
    "Purchase_Frequency",
    "Avg_Browsing_Time",
    "Total_Engagement_Score",
    # RFMB-C 12 核心特征（Min-Max 标准化后）
    "R_Recency_Days_MM",
    "R_Avg_Interval_Days_MM",
    "F_Orders_30d_MM",
    "F_Orders_90d_MM",
    "M_Sales_30d_MM",
    "M_AOV_30d_MM",
    "M_Sales_Var_90d_MM",
    "B_Browse_Count_30d_MM",
    "B_Avg_Browse_Time_30d_MM",
    "B_Depth_30d_MM",
    "C_Top1_Ratio_MM",
    "C_Top3_Coverage_MM",
}

# 日期列（会被解析为 datetime.date/datetime，并写入 DATE/DATETIME 类型）
_DATE_COLS = {
    "Ref_Date",
    "Last_Purchase_Date",
}


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
    if "Customer_ID" not in df.columns:
        print("错误：CSV 缺少 Customer_ID 列，无法作为主键导入。")
        return

    db_name = MYSQL_CONFIG["database"]
    # 先连接不含 database 的实例以创建库
    config_no_db = {k: v for k, v in MYSQL_CONFIG.items() if k != "database"}
    # 使用上下文管理器确保连接与游标可靠关闭
    with pymysql.connect(**config_no_db) as conn:
        with conn.cursor() as cur:
            # 1) 创建数据库并切换
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cur.execute(f"USE `{db_name}`")

            # 2) 强类型建表（主键 + 索引），可重复执行
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                    `Customer_ID` VARCHAR(64) NOT NULL,
                    `Customer_Name` VARCHAR(255) NULL,
                    `Total_Orders` INT NULL,
                    `Total_Sales` DOUBLE NULL,
                    `Avg_Order_Value` DOUBLE NULL,
                    `Customer_Lifetime_Days` INT NULL,
                    `Days_Since_Last_Purchase` INT NULL,
                    `Ref_Date` DATE NULL,
                    `Last_Purchase_Date` DATE NULL,
                    `Purchase_Frequency` DOUBLE NULL,
                    `Avg_Browsing_Time` DOUBLE NULL,
                    `Unique_Products_Purchased` INT NULL,
                    `Favorite_Product` VARCHAR(255) NULL,
                    `Total_Likes` INT NULL,
                    `Total_Shares` INT NULL,
                    `Total_Add_to_Cart` INT NULL,
                    `Total_Engagement_Score` DOUBLE NULL,
                    `R_Recency_Days_MM` DOUBLE NULL,
                    `R_Avg_Interval_Days_MM` DOUBLE NULL,
                    `F_Orders_30d_MM` DOUBLE NULL,
                    `F_Orders_90d_MM` DOUBLE NULL,
                    `M_Sales_30d_MM` DOUBLE NULL,
                    `M_AOV_30d_MM` DOUBLE NULL,
                    `M_Sales_Var_90d_MM` DOUBLE NULL,
                    `B_Browse_Count_30d_MM` DOUBLE NULL,
                    `B_Avg_Browse_Time_30d_MM` DOUBLE NULL,
                    `B_Depth_30d_MM` DOUBLE NULL,
                    `C_Top1_Ratio_MM` DOUBLE NULL,
                    `C_Top3_Coverage_MM` DOUBLE NULL,
                    `Cluster` INT NULL,
                    `Cluster_Label` VARCHAR(64) NULL,
                    PRIMARY KEY (`Customer_ID`),
                    KEY `idx_cluster` (`Cluster`),
                    KEY `idx_cluster_label` (`Cluster_Label`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            if RESET_TABLE:
                cur.execute(f"TRUNCATE TABLE `{TABLE_NAME}`")

            # 3) 统一类型：确保数值列进入 MySQL 时仍是数值，避免排序/聚合变成字符串语义
            for c in df.columns:
                if c in _INT_COLS:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                elif c in _DATE_COLS:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                elif c in _FLOAT_COLS:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    if c in df.columns:
                        df[c] = df[c].where(~pd.isna(df[c]), None)

            cols = df.columns.tolist()
            placeholders = ", ".join(["%s"] * len(cols))
            col_list_sql = ", ".join([f"`{c}`" for c in cols])
            update_sql = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in cols if c != "Customer_ID"])
            sql = (
                f"INSERT INTO `{TABLE_NAME}` ({col_list_sql}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {update_sql}"
            )

            params = []
            for _, row in df.iterrows():
                vals = []
                for c in cols:
                    v = row[c]
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        vals.append(None)
                    elif c in _INT_COLS:
                        vals.append(int(v))
                    elif c in _DATE_COLS:
                        # pandas Timestamp -> python date
                        try:
                            vals.append(v.date() if hasattr(v, "date") else v)
                        except Exception:
                            vals.append(None)
                    elif c in _FLOAT_COLS:
                        vals.append(float(v))
                    else:
                        vals.append(str(v))
                params.append(tuple(vals))

            if params:
                cur.executemany(sql, params)

            conn.commit()
            cur.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`")
            cnt = cur.fetchone()[0]
            print(f"导入完成，共 {cnt} 行，表名：{TABLE_NAME}")


if __name__ == "__main__":
    main()
