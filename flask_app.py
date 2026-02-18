"""
基于 everyuser 画像接口的简单 Flask 网页

用法：
1. 在当前目录下确保有 `customer_clusters_simple.csv` 数据文件。
2. 在终端中安装依赖（如有需要）：
     pip install flask pandas numpy
3. 运行本服务：
     python flask_app.py
4. 在浏览器打开：http://127.0.0.1:5000
"""

from __future__ import annotations

from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np


app = Flask(__name__)

DATA_CSV_PATH = "customer_clusters_simple.csv"


# === 直接复用 everyuser 中的数据接口逻辑 ===

def load_data(path: str = DATA_CSV_PATH) -> pd.DataFrame:
    """从 CSV 加载用户特征数据并返回 DataFrame。"""
    df = pd.read_csv(path)
    return df


def create_individual_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """将原始特征表标准化为统一的用户画像表格。"""
    profiles = []

    for _, row in df.iterrows():
        # 只在值为 None 或 NaN 时才使用默认值，避免把 0 当成缺失
        def _safe_float(v, default: float = 0.0) -> float:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return float(default)
            return float(v)

        def _safe_int(v, default: int = 0) -> int:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return int(default)
            return int(v)

        total_sales = _safe_float(row.get("Total_Sales", 0.0), 0.0)
        total_orders = _safe_int(row.get("Total_Orders", 0), 0)
        avg_order_value = _safe_float(row.get("Avg_Order_Value", 0.0), 0.0)
        lifetime_val = row.get("Customer_Lifetime_Days", row.get("Customer_Lifetime", 0))
        customer_lifetime = _safe_float(lifetime_val, 0.0)
        purchase_freq = _safe_float(row.get("Purchase_Frequency", 0.0), 0.0)
        profit_margin = _safe_float(row.get("Profit_Margin", 0.0), 0.0)

        last_purchase_raw = row.get(
            "Days_Since_Last_Purchase", row.get("Last_Purchase_Days_Ago", np.nan)
        )
        if last_purchase_raw is None or (
            isinstance(last_purchase_raw, float) and np.isnan(last_purchase_raw)
        ):
            last_purchase_days_ago = np.nan
        else:
            last_purchase_days_ago = float(last_purchase_raw)

        engagement_raw = row.get(
            "Total_Engagement_Score", row.get("Engagement_Score", 0.0)
        )
        engagement_score = _safe_float(engagement_raw, 0.0)

        unique_products_raw = row.get(
            "Unique_Products_Purchased", row.get("Unique_Products_Bought", 0)
        )
        unique_products_bought = _safe_int(unique_products_raw, 0)

        profile = {
            "Customer_ID": row.get("Customer ID", row.get("Customer_ID", None)),
            "Customer_Name": row.get(
                "Customer Name", row.get("Customer_Name", "Unknown")
            ),
            "Cluster": row.get("Cluster", None),
            "Total_Sales": total_sales,
            "Total_Orders": total_orders,
            "Avg_Order_Value": avg_order_value,
            "Customer_Lifetime": customer_lifetime,
            "Purchase_Frequency": purchase_freq,
            "Profit_Margin": profit_margin,
            "Last_Purchase_Days_Ago": last_purchase_days_ago,
            "Engagement_Score": engagement_score,
            "Segment": row.get("Segment", None),
            "Region": row.get("Region", None),
            "Country": row.get("Country", None),
            "Gender": row.get("Gender", None),
            "Age": row.get("Age", np.nan),
            "Education": row.get("Education", None),
            "Marital_Status": row.get(
                "Marital Status", row.get("Marital_Status", None)
            ),
            "Favorite_Product": row.get("Favorite_Product", "Unknown"),
            "Unique_Products_Bought": unique_products_bought,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


def calculate_percentiles_and_rankings(
    df: pd.DataFrame, metrics: list[str] | None = None
) -> pd.DataFrame:
    """为指定指标计算百分位、排名、评级和与均值对比。"""
    result = df.copy()
    if metrics is None:
        metrics = [
            "Total_Sales",
            "Total_Orders",
            "Avg_Order_Value",
            "Purchase_Frequency",
            "Profit_Margin",
            "Engagement_Score",
            "Customer_Lifetime",
        ]

    def get_grade(percentile: float) -> str:
        if percentile >= 80:
            return "A"
        elif percentile >= 60:
            return "B"
        elif percentile >= 40:
            return "C"
        elif percentile >= 20:
            return "D"
        else:
            return "E"

    for metric in metrics:
        if metric in result.columns:
            result[f"{metric}_Percentile"] = result[metric].rank(pct=True) * 100
            result[f"{metric}_Rank"] = result[metric].rank(
                ascending=False, method="min"
            )
            result[f"{metric}_Grade"] = result[f"{metric}_Percentile"].apply(get_grade)
            overall_mean = result[metric].mean()
            if overall_mean != 0 and not np.isnan(overall_mean):
                result[f"{metric}_vs_Mean"] = (
                    (result[metric] - overall_mean) / overall_mean * 100
                ).round(1)
            else:
                result[f"{metric}_vs_Mean"] = 0.0

    return result


def get_profiles_scored() -> pd.DataFrame:
    """从 CSV 读取并返回带评分的用户画像 DataFrame。"""
    raw = load_data(DATA_CSV_PATH)
    profiles = create_individual_profiles(raw)
    profiles_scored = calculate_percentiles_and_rankings(profiles)
    return profiles_scored


# === Flask 路由 ===

INDEX_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>客户画像仪表盘（Flask）</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #ddd; padding: 4px 6px; }
      th { background-color: #f5f5f5; }
      .summary { margin-bottom: 16px; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; background: #eee; margin-right: 4px; }
      .search-box { margin-bottom: 12px; }
    </style>
  </head>
  <body>
    <h1>客户画像仪表盘</h1>

    <div class="summary">
      <p>总用户数：{{ total_users }}</p>
      <p>聚类分布：</p>
      {% for cluster, count in cluster_counts.items() %}
        <span class="badge">{{ cluster }}：{{ count }}</span>
      {% endfor %}
    </div>

    <div class="search-box">
      <form method="get" action="/">
        <label>按客户 ID 过滤：</label>
        <input type="text" name="customer_id" value="{{ customer_id or '' }}" />
        <label>按聚类：</label>
        <input type="text" name="cluster" value="{{ cluster or '' }}" />
        <button type="submit">筛选</button>
      </form>
    </div>

    {{ table_html | safe }}
  </body>
  </html>
"""


@app.route("/")
def index():
    df = get_profiles_scored()

    # 简单筛选
    customer_id = request.args.get("customer_id", "").strip() or None
    cluster = request.args.get("cluster", "").strip() or None

    if customer_id:
        df = df[df["Customer_ID"].astype(str).str.contains(customer_id, case=False)]
    if cluster:
        # 既支持按数值聚类编号筛选，也支持按中文标签筛选
        df = df[
            (df.get("Cluster").astype(str) == cluster)
            | (df.get("Cluster_Label").astype(str) == cluster)
        ]

    total_users = len(df)

    # 优先按中文标签统计分布，若不存在则退回数值型 Cluster
    if "Cluster_Label" in df.columns:
        cluster_key_col = "Cluster_Label"
    else:
        cluster_key_col = "Cluster" if "Cluster" in df.columns else None

    if cluster_key_col is not None:
        cluster_counts = (
            df[cluster_key_col].value_counts(dropna=False).sort_index().to_dict()
        )
    else:
        cluster_counts = {}

    # 只展示前若干列，避免表太宽
    display_cols = [
        "Customer_ID",
        "Customer_Name",
        "Cluster_Label",
        "Cluster",
        "Total_Sales",
        "Total_Orders",
        "Avg_Order_Value",
        "Customer_Lifetime",
        "Purchase_Frequency",
        "Profit_Margin",
        "Last_Purchase_Days_Ago",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    table_html = df[existing_cols].head(200).to_html(index=False)

    return render_template_string(
        INDEX_TEMPLATE,
        total_users=total_users,
        cluster_counts=cluster_counts,
        table_html=table_html,
        customer_id=customer_id,
        cluster=cluster,
    )


if __name__ == "__main__":
    # debug=True 方便开发时自动重载
    app.run(host="127.0.0.1", port=5000, debug=True)

