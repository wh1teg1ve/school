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

import os
from urllib.parse import quote

from flask import Flask, jsonify, redirect, render_template_string, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go


app = Flask(__name__)

DATA_CSV_PATH = "customer_clusters_simple.csv"
# 是否启用 MySQL 数据源（否则默认用 CSV），由环境变量控制
USE_MYSQL = os.environ.get("USE_MYSQL", "").lower() in ("1", "true", "yes")

# MySQL 连接配置，支持通过环境变量覆盖
MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "customer_profile"),
    "charset": "utf8mb4",
}
# 使用的数据表名
MYSQL_TABLE = os.environ.get("MYSQL_TABLE", "customer_clusters")


def _load_from_csv(path: str = DATA_CSV_PATH) -> pd.DataFrame:
    """从 CSV 加载用户特征数据。"""
    return pd.read_csv(path)


def _load_from_mysql() -> pd.DataFrame:
    """从 MySQL 加载用户特征数据。"""
    import pymysql

    conn = pymysql.connect(**MYSQL_CONFIG)
    try:
        df = pd.read_sql(f"SELECT * FROM `{MYSQL_TABLE}`", conn)
        return df
    finally:
        conn.close()


def load_data(path: str = DATA_CSV_PATH) -> pd.DataFrame:
    """根据配置从 CSV 或 MySQL 加载用户特征数据并返回 DataFrame。"""
    if USE_MYSQL:
        return _load_from_mysql()
    return _load_from_csv(path)


def create_individual_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """将原始特征表标准化为统一的用户画像表格。"""
    profiles = []

    # 遍历原始数据的每一行，将字段名和缺失值统一为内部标准格式
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
            "Cluster_Label": row.get("Cluster_Label", None),
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

    # 针对每个指标分别计算百分位、排名、等级、与均值对比
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
    """从 CSV 或 MySQL 读取并返回带评分的用户画像 DataFrame。"""
    raw = load_data()
    # 1) 统一字段与缺失值
    profiles = create_individual_profiles(raw)
    # 2) 计算各类指标的百分位和排名
    profiles_scored = calculate_percentiles_and_rankings(profiles)
    return profiles_scored


# === Flask 路由 ===

INDEX_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>客户画像仪表盘（Flask）</title>
    <style>
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px; background: #f8f9fa; }
      .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
      h1 { margin: 0 0 20px; font-size: 1.5em; color: #1a1a1a; }
      .summary { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-bottom: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px; }
      .summary p { margin: 0; }
      .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; background: #e3f2fd; color: #1565c0; margin-right: 6px; margin-bottom: 4px; font-size: 0.9em; text-decoration: none; }
      .badge:hover { background: #bbdefb; color: #0d47a1; }
      .badge-active { background: #1976d2; color: #fff; }
      .badge-active:hover { background: #1565c0; color: #fff; }
      .badge-clear { background: #ffebee; color: #c62828; }
      .badge-clear:hover { background: #ffcdd2; color: #b71c1c; }
      .search-form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; padding: 16px; background: #f8f9fa; border-radius: 8px; }
      .search-form label { font-weight: 500; }
      .search-form input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
      .search-form select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; background: #fff; }
      .search-form button { padding: 8px 20px; background: #2196f3; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
      .search-form button:hover { background: #1976d2; }
      .btn-random { display: inline-block; padding: 8px 20px; background: #2196f3; color: #fff; border-radius: 6px; font-size: 14px; }
      .btn-random:hover { background: #1976d2; color: #fff; text-decoration: none; }
      .compare-form { margin-bottom: 24px; }
      .compare-actions-top, .compare-actions-bottom { display: flex; justify-content: space-between; align-items: center; margin: 4px 0; font-size: 0.9em; color: #666; }
      .btn-compare-selected { padding: 6px 14px; background: #4caf50; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
      .btn-compare-selected:hover { background: #43a047; }
      .checkbox-col { width: 56px; text-align: center; }
      .btn-page { display: inline-block; padding: 6px 14px; background: #2196f3; color: #fff; border-radius: 6px; font-size: 13px; text-decoration: none; margin-left: 6px; }
      .btn-page:hover { background: #1976d2; color: #fff; text-decoration: none; }
      .table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid #eee; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #eee; padding: 10px 12px; }
      th { background: #f5f5f5; font-weight: 600; }
      tr:hover { background: #fafafa; }
      a { color: #1976d2; text-decoration: none; }
      a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>客户画像仪表盘</h1>

      <div class="summary">
        <p>
          <strong>总用户数：</strong>{{ total_users }}；
          <strong>总消费金额：</strong>¥{{ "%.0f"|format(overall_sales) }}；
          <strong>总订单数：</strong>{{ "%.0f"|format(overall_orders) }}；
          <strong>整体用户平均订单价格：</strong>¥{{ "%.2f"|format(overall_aov) }}
        </p>
        <p>
          <strong>高价值客户数：</strong>{{ high_value_count }}；
          <strong>潜在流失客户数：</strong>{{ churn_count }}
        </p>
        <p><strong>聚类分布：</strong> 点击筛选
        {% for name, count, href in cluster_badges %}
          <a href="{{ href }}" class="badge {{ 'badge-active' if cluster == name else '' }}">{{ name }}：{{ count }}</a>
        {% endfor %}
        {% if cluster %}<a href="/" class="badge badge-clear">清除筛选</a>{% endif %}
        </p>
      </div>

      <div class="section" style="margin-bottom: 20px; padding: 16px; background:#fff; border-radius: 10px; box-shadow:0 1px 4px rgba(0,0,0,.04);">
        <h2 style="font-size:1.05em;margin:0 0 8px;">各客户群体用户占比</h2>
        <p style="color:#666;font-size:0.88em;margin:0 0 10px;">帮助快速对比不同客户群体的价值水平。</p>
        {{ cluster_bar_html | safe }}
      </div>

      <form class="search-form" method="get" action="/">
        <label>客户 ID：</label>
        <input type="text" name="customer_id" value="{{ customer_id or '' }}" placeholder="筛选客户ID" />
        <label>聚类：</label>
        <input type="text" name="cluster" value="{{ cluster or '' }}" placeholder="按群体筛选" />
        <button type="submit">筛选</button>
        <a href="/user/random" class="btn-random">随机客户</a>
      </form>

      <form class="compare-form" method="get" action="/compare">
        <div class="compare-actions-top">
          <span>在下方勾选多个客户后，可一键进入对比页面（跨筛选和分页将自动记住）。当前第 {{ page }} / {{ total_pages }} 页，每页 {{ per_page }} 条，合计 {{ total_users }} 个客户。<span id="selected-counter"></span></span>
          <button type="submit" class="btn-compare-selected">对比选中客户</button>
        </div>
        <div class="table-wrap">
          {{ table_html | safe }}
        </div>
        <div class="compare-actions-bottom">
          <div></div>
          <div>
            {% if page > 1 %}
              <a class="btn-page" href="/?page={{ page - 1 }}&per_page={{ per_page }}&customer_id={{ (customer_id or '') | urlencode }}&cluster={{ (cluster or '') | urlencode }}&sort_by={{ sort_by }}&sort_dir={{ sort_dir }}">上一页</a>
            {% endif %}
            {% if page < total_pages %}
              <a class="btn-page" href="/?page={{ page + 1 }}&per_page={{ per_page }}&customer_id={{ (customer_id or '') | urlencode }}&cluster={{ (cluster or '') | urlencode }}&sort_by={{ sort_by }}&sort_dir={{ sort_dir }}">下一页</a>
            {% endif %}
          </div>
        </div>
      </form>
      <script>
        (function() {
          const STORAGE_KEY = 'selectedCustomerIds';

          function loadSelected() {
            try {
              const raw = localStorage.getItem(STORAGE_KEY);
              if (!raw) return [];
              const arr = JSON.parse(raw);
              return Array.isArray(arr) ? arr : [];
            } catch (e) {
              return [];
            }
          }

          function saveSelected(ids) {
            try {
              localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
            } catch (e) {}
          }

          function updateCounter(ids) {
            const el = document.getElementById('selected-counter');
            if (!el) return;
            if (!ids.length) {
              el.textContent = '（已选 0 位客户）';
            } else {
              el.textContent = '（已选 ' + ids.length + ' 位客户，将在对比页展示）';
            }
          }

          document.addEventListener('DOMContentLoaded', function() {
            const checkboxes = Array.from(document.querySelectorAll('input[name="ids"][type="checkbox"]'));
            const form = document.querySelector('.compare-form');
            let selected = loadSelected();

            // 初始化勾选状态
            checkboxes.forEach(cb => {
              if (selected.includes(cb.value)) {
                cb.checked = true;
              }
            });
            updateCounter(selected);

            // 勾选/取消时更新本地存储
            checkboxes.forEach(cb => {
              cb.addEventListener('change', function() {
                const id = this.value;
                if (this.checked) {
                  if (!selected.includes(id)) {
                    selected.push(id);
                  }
                } else {
                  selected = selected.filter(x => x !== id);
                }
                saveSelected(selected);
                updateCounter(selected);
              });
            });

            // 提交到 /compare 时，把所有已选 ID 作为隐藏字段带上（包括不在当前页的）
            if (form) {
              form.addEventListener('submit', function() {
                Array.from(form.querySelectorAll('input[type="hidden"][name="ids"]')).forEach(el => el.remove());
                selected.forEach(id => {
                  const hidden = document.createElement('input');
                  hidden.type = 'hidden';
                  hidden.name = 'ids';
                  hidden.value = id;
                  form.appendChild(hidden);
                });
              });
            }
          });
        })();
      </script>
    </div>
  </body>
  </html>
"""


# 客户画像报告页模板（含百分位可视化）
USER_PROFILE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>客户画像报告 - {{ customer_name or customer_id }}</title>
    <style>
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px; background: #f8f9fa; }
      .container { max-width: 1100px; margin: 0 auto; }
      .back { margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
      .back a { color: #1976d2; text-decoration: none; font-size: 0.95em; }
      .back a:hover { text-decoration: underline; }
      .btn-compare { display: inline-block; padding: 4px 10px; border-radius: 6px; background: #4caf50; color: #fff; font-size: 0.9em; }
      .btn-compare:hover { background: #388e3c; color: #fff; text-decoration: none; }
      h1 { margin: 0 0 20px; font-size: 1.5em; color: #1a1a1a; }
      .section { margin-bottom: 24px; padding: 20px; background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
      .section h2 { font-size: 1.1em; margin: 0 0 8px; color: #333; }
      .section-desc { color: #666; font-size: 0.9em; margin-bottom: 14px; }
      .headline { font-size: 0.95em; color: #555; margin: -8px 0 16px; }
      .header-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
      .header-card { padding: 12px 16px; background: #f8f9fa; border-radius: 8px; }
      .header-card strong { display: block; font-size: 0.8em; color: #666; margin-bottom: 4px; }
      .metric-row { display: grid; grid-template-columns: 140px 120px 180px auto; align-items: center; column-gap: 12px; margin: 10px 0; }
      .metric-row .label { color: #555; }
      .metric-row .value { font-weight: 600; }
      .metric-row .percent-bar { position: relative; width: 100%; height: 14px; background: rgba(33,150,243,0.18); border-radius: 7px; overflow: hidden; }
      .metric-row .percent-marker { position: absolute; top: 0; bottom: 0; width: 2px; background: #e91e63; transform: translateX(-1px); }
      .metric-row .percent-label { font-size: 0.8em; color: #555; font-weight: 600; margin-left: 8px; }
      .metric-row .pct-text { font-size: 0.9em; color: #666; text-align: right; }
      .chart-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }
      .chart-row > div { min-width: 0; }
      .detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px 24px; }
      .detail-grid p { margin: 8px 0; padding: 0; font-size: 0.95em; }
      .detail-grid strong { color: #555; font-weight: 500; }
      .analysis-list { margin: 0; padding-left: 20px; line-height: 1.7; color: #333; }
      .analysis-list li { margin: 8px 0; }
      .rec-list { display: flex; flex-direction: column; gap: 12px; }
      .rec-card { display: flex; gap: 12px; padding: 14px 16px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #2196f3; }
      .rec-priority { flex-shrink: 0; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; background: #2196f3; color: #fff; border-radius: 6px; font-size: 0.85em; font-weight: 600; }
      .rec-card div p { margin: 6px 0 0; font-size: 0.92em; color: #555; line-height: 1.5; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="back">
        <a href="/">&larr; 返回用户列表</a>
        <a href="/compare?ids={{ customer_id }}" class="btn-compare">对比多个客户</a>
      </div>
      <h1>客户画像报告</h1>
      {% if headline_summary %}
      <p class="headline">{{ headline_summary }}</p>
      {% endif %}

      <div class="section">
        <h2>基本信息</h2>
        <div class="header-grid">
          <div class="header-card"><strong>客户 ID</strong>{{ customer_id }}</div>
          <div class="header-card"><strong>客户姓名</strong>{{ customer_name or '-' }}</div>
          <div class="header-card"><strong>所属群体</strong>{{ cluster_label or cluster or '-' }}</div>
          <div class="header-card"><strong>偏好产品</strong>{{ favorite_product or '-' }}</div>
          <div class="header-card"><strong>综合评分</strong>{{ overall_score }}</div>
          <div class="header-card"><strong>运营优先级</strong>{{ priority_label }}</div>
        </div>
      </div>

      <div class="section">
        <h2>指标排名</h2>
        <p class="section-desc">百分位越高表示该指标优于越多用户。条形图表示所处百分位水平。</p>
        {% for m in metrics %}
        <div class="metric-row">
          <span class="label">{{ m.label }}</span>
          <span class="value">{{ m.value }}</span>
          <span class="percent-bar">
            <span class="percent-marker" style="left:{{ m.marker_pos }}%;"></span>
          </span>
          {% set p = m.percentile | round(0) | int %}
          <span class="pct-text">
            位于前 {{ p }}% · 超过 {{ 100 - p }}% 用户
          </span>
        </div>
        {% endfor %}
      </div>

      <div class="section">
        <h2>雷达图：各维度百分位对比</h2>
        <p class="section-desc">该客户在各指标上相对于全体用户的百分位（0–100）。</p>
        <div style="max-width: 500px; margin: 0 auto;">
          {{ radar_chart_html | safe }}
        </div>
      </div>

      <div class="section">
        <h2>散点图：在全体用户中的位置</h2>
        <p class="section-desc">灰色点为其他用户，<span style="color:#e91e63;font-weight:bold;">红色菱形</span>为当前客户。</p>
        <div class="chart-row">
          <div>{{ scatter_sales_orders_html | safe }}</div>
          <div>{{ scatter_avg_freq_html | safe }}</div>
        </div>
      </div>

      <div class="section">
        <h2>订单趋势</h2>
        <p class="section-desc">近 12 个月订单数与全体平均对比。</p>
        {{ orders_trend_html | safe }}
      </div>

      <div class="section">
        <h2>客户分析</h2>
        <p class="section-desc">基于 RFM-BC 聚类与指标百分位的综合分析。</p>
        <ul class="analysis-list">
          {% for item in analysis_items %}
          <li><strong>{{ item.label }}：</strong>{{ item.text }}</li>
          {% endfor %}
        </ul>
      </div>

      <div class="section">
        <h2>推荐营销方案</h2>
        <p class="section-desc">根据客户群体特征推荐的营销策略，按优先级排序。</p>
        <div class="rec-list">
          {% for rec in marketing_recs %}
          <div class="rec-card">
            <span class="rec-priority">P{{ rec.priority }}</span>
            <div>
              <strong>{{ rec.title }}</strong>
              <p>{{ rec.desc }}</p>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <div class="section">
        <h2>详细指标</h2>
        <div class="detail-grid">
          <p><strong>总销售额：</strong>{{ total_sales }}</p>
          <p><strong>订单数：</strong>{{ total_orders }}</p>
          <p><strong>用户平均订单价格：</strong>{{ avg_order_value }}</p>
          <p><strong>客户生命周期：</strong>{{ customer_lifetime }} 天</p>
          <p><strong>购买频率：</strong>{{ purchase_frequency }}</p>
          <p><strong>利润率：</strong>{{ profit_margin }}</p>
          <p><strong>距上次购买：</strong>{{ last_purchase_days }} 天</p>
          <p><strong>互动分：</strong>{{ engagement_score }}</p>
        </div>
      </div>
    </div>
  </body>
  </html>
"""


COMPARE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>多客户对比 - 客户画像</title>
    <style>
      * { box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px; background: #f8f9fa; }
      .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 28px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
      h1 { margin: 0 0 20px; font-size: 1.5em; color: #1a1a1a; }
      .back { margin-bottom: 16px; }
      .back a { color: #1976d2; text-decoration: none; font-size: 0.95em; }
      .back a:hover { text-decoration: underline; }
      .section { margin-bottom: 20px; }
      .section h2 { font-size: 1.1em; margin: 0 0 8px; color: #333; }
      .section-desc { color: #666; font-size: 0.9em; margin-bottom: 10px; }
      .compare-form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 14px 16px; background: #f8f9fa; border-radius: 8px; margin-bottom: 18px; }
      .compare-form label { font-weight: 500; }
      .compare-form input { flex: 1; min-width: 260px; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
      .compare-form button { padding: 8px 18px; background: #2196f3; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
      .compare-form button:hover { background: #1976d2; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }
      th, td { border: 1px solid #eee; padding: 8px 10px; text-align: center; }
      th { background: #f5f5f5; font-weight: 600; }
      tbody tr:nth-child(even) { background: #fafafa; }
      .metric-name { text-align: left; white-space: nowrap; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; background: #e3f2fd; color: #1565c0; }
      .empty-hint { color: #999; font-size: 0.9em; margin-top: 8px; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="back"><a href="/">&larr; 返回仪表盘</a></div>
      <h1>多客户画像对比</h1>

      <div class="section">
        <h2>选择客户</h2>
        <p class="section-desc">在下方输入多个客户 ID，使用逗号或空格分隔，例如：AB-00363, AB-00421, AH-00722，或直接点击“随机选择客户”。</p>
        <form class="compare-form" method="get" action="/compare">
          <label for="ids">客户 ID 列表：</label>
          <input id="ids" type="text" name="ids" value="{{ ids_text or '' }}" placeholder="例如：AB-00363, AB-00421, AH-00722" />
          <button type="submit">对比</button>
          <button type="submit" name="random" value="1">保留第一位 + 随机选择后两位</button>
        </form>
        {% if not rows %}
          <p class="empty-hint">当前暂无对比结果，请在上方输入至少一个有效的客户 ID。</p>
        {% endif %}
      </div>

      {% if rows %}
      <div class="section">
        <h2>基本信息对比</h2>
        <table>
          <thead>
            <tr>
              <th>客户 ID</th>
              <th>姓名</th>
              <th>群体</th>
            </tr>
          </thead>
          <tbody>
            {% for r in rows %}
            <tr>
              <td><a href="/user/{{ r.customer_id }}">{{ r.customer_id }}</a></td>
              <td>{{ r.customer_name or '-' }}</td>
              <td><span class="pill">{{ r.cluster_label or r.cluster or '-' }}</span></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>关键指标对比</h2>
        <table>
          <thead>
            <tr>
              <th class="metric-name">指标</th>
              {% for r in rows %}
              <th>{{ r.customer_id }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for m in metrics %}
            <tr>
              <td class="metric-name">{{ m.label }}</td>
              {% for r in rows %}
              <td>{{ r.metrics.get(m.key, '-') }}</td>
              {% endfor %}
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="section">
        <h2>雷达图对比</h2>
        <p class="section-desc">展示所选客户在各指标上的百分位对比（0–100）。</p>
        {{ radar_chart_html | safe }}
      </div>
      {% endif %}
    </div>
  </body>
  </html>
"""


def _safe(val):
    """将 NaN/None 转为可显示字符串。"""
    if val is None or pd.isna(val):
        return "-"
    if isinstance(val, (float, np.floating)):
        return f"{val:,.2f}"
    return str(val)


def _compute_recent_percentile(df: pd.DataFrame, customer_id: str) -> float:
    """
    将 Last_Purchase_Days_Ago 转为“越近越高”的百分位（0-100）。
    若缺失则返回 50。
    """
    col = "Last_Purchase_Days_Ago"
    if col not in df.columns:
        return 50.0
    s = df[col].copy()
    s = s.fillna(s.median() if pd.notna(s.median()) else 0)
    # 越小越好：先算“越小越靠前”的 pct，再取 100 - pct
    pct = s.rank(pct=True, ascending=True) * 100
    recent = 100.0 - float(pct[df["Customer_ID"].astype(str) == str(customer_id)].iloc[0])
    return max(0.0, min(100.0, recent))


def _add_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    为首页添加活跃度分数与分档（高/中/低）。
    - Activity_Score: 0-100（越大越活跃）
    - Activity_Level: 高/中/低
    说明：使用“最近购买（越近越高）”与“购买频率百分位”加权。
    """
    result = df.copy()
    # 先找到“距上次购买天数”这一列（不同数据源可能是不同名字）
    if "Last_Purchase_Days_Ago" in result.columns:
        s = result["Last_Purchase_Days_Ago"].copy()
    elif "Days_Since_Last_Purchase" in result.columns:
        s = result["Days_Since_Last_Purchase"].copy()
        result["Last_Purchase_Days_Ago"] = s
    else:
        s = pd.Series([np.nan] * len(result), index=result.index)
        result["Last_Purchase_Days_Ago"] = s

    # 近购百分位：天数越小越好 -> recent_pct 越高
    s_filled = s.fillna(s.median() if pd.notna(s.median()) else 0)
    recent_pct = 100.0 - (s_filled.rank(pct=True, ascending=True) * 100.0)

    # 购买频率百分位：若没有该列，则默认 50 分
    freq_pct = (
        result["Purchase_Frequency_Percentile"]
        if "Purchase_Frequency_Percentile" in result.columns
        else pd.Series([50.0] * len(result), index=result.index)
    )
    freq_pct = freq_pct.fillna(50.0).astype(float)

    # 活跃度分数 = 0.7 * “最近购买百分位” + 0.3 * “购买频率百分位”
    activity_score = (0.7 * recent_pct.astype(float) + 0.3 * freq_pct).clip(0, 100)
    result["Activity_Score"] = activity_score.round(1)

    def _level(v: float) -> str:
        if v >= 70:
            return "高"
        if v >= 40:
            return "中"
        return "低"

    # 活跃档位（高/中/低），方便在表格中快速阅读
    result["Activity_Level"] = result["Activity_Score"].apply(_level)
    return result


def _compute_overall_score(df: pd.DataFrame, r: pd.Series, customer_id: str) -> float:
    """综合评分（0-100），基于多个百分位指标加权。"""
    weights = {
        "Total_Sales_Percentile": 0.25,
        "Total_Orders_Percentile": 0.20,
        "Avg_Order_Value_Percentile": 0.20,
        "Engagement_Score_Percentile": 0.15,
        "Customer_Lifetime_Percentile": 0.10,
        "Recent_Percentile": 0.10,
    }

    # 取某个百分位列的值，缺失则给默认 50 分
    def _get_pct(col: str) -> float:
        v = r.get(col)
        if v is None or pd.isna(v):
            return 50.0
        return float(v)

    # 最近活跃度单独算（基于 Last_Purchase_Days_Ago）
    recent_p = _compute_recent_percentile(df, customer_id)
    score = (
        weights["Total_Sales_Percentile"] * _get_pct("Total_Sales_Percentile")
        + weights["Total_Orders_Percentile"] * _get_pct("Total_Orders_Percentile")
        + weights["Avg_Order_Value_Percentile"] * _get_pct("Avg_Order_Value_Percentile")
        + weights["Engagement_Score_Percentile"] * _get_pct("Engagement_Score_Percentile")
        + weights["Customer_Lifetime_Percentile"] * _get_pct("Customer_Lifetime_Percentile")
        + weights["Recent_Percentile"] * recent_p
    )
    return max(0.0, min(100.0, score))


def _compute_priority_label(cluster_label: str, score: float, last_days) -> str:
    """运营优先级：高/中/低（含简短说明）。"""
    label = str(cluster_label or "")
    days = None
    if last_days is not None and not pd.isna(last_days):
        try:
            days = float(last_days)
        except Exception:
            days = None

    # 先按综合评分给一个大致档位，再用群体+最近购买时间做细分
    if score >= 80:
        return "高（重点维护）"
    if "高价值" in label and (days is not None and days > 60):
        return "高（重点挽留）"
    if score >= 50:
        return "中（重点挖掘）"
    return "低（常规关注）"


@app.route("/compare")
def compare_customers():
    """多客户对比页面。通过 ids 参数（逗号/空格分隔）选择多个客户进行关键指标对比。"""
    df = get_profiles_scored()

    # 支持两种输入：
    # 1) ids=AB-1,AB-2（文本输入框）
    # 2) ids=AB-1&ids=AB-2（首页复选框多选）
    ids_text = (request.args.get("ids", "") or "").strip()

    # 统一把 ids 解析成列表 id_list
    raw_list = [x for x in request.args.getlist("ids") if x and str(x).strip()]
    if len(raw_list) > 1:
        # 来自首页多选：ids=ID1&ids=ID2...
        id_list = [str(x).strip() for x in raw_list]
    elif len(raw_list) == 1:
        # 来自输入框：ids="ID1, ID2" 或 "ID1 ID2"
        raw = str(raw_list[0]).strip()
        parts = raw.replace("，", ",").replace(" ", ",").split(",")
        id_list = [p.strip() for p in parts if p.strip()]
    else:
        id_list = []

    # 兼容极少数情况下 ids_text 里有值但 getlist 为空
    if not id_list and ids_text:
        parts = ids_text.replace("，", ",").replace(" ", ",").split(",")
        id_list = [p.strip() for p in parts if p.strip()]

    # 如果点击了“随机选择”按钮，则按“保留第一位 + 随机补齐”的规则抽样
    random_flag = request.args.get("random")
    if random_flag:
        if df.empty:
            subset = df.iloc[0:0].copy()
            id_list = []
            ids_text = ""
        else:
            try:
                n = int(request.args.get("n", "3"))
            except ValueError:
                n = 3
            n = max(1, min(n, len(df)))

            # 若已输入/已选客户列表，则保留第一位，其余随机补齐
            keep_id = id_list[0] if id_list else None
            if keep_id and (df["Customer_ID"].astype(str) == str(keep_id)).any():
                base_row = df[df["Customer_ID"].astype(str) == str(keep_id)].head(1)
                pool = df[df["Customer_ID"].astype(str) != str(keep_id)]
                need = max(0, min(n - 1, len(pool)))
                sampled = pool.sample(n=need, random_state=None) if need > 0 else pool.iloc[0:0]
                subset = pd.concat([base_row, sampled], ignore_index=True)
            else:
                subset = df.sample(n=n, random_state=None)

            id_list = subset["Customer_ID"].astype(str).tolist()
            ids_text = ", ".join(id_list)
    else:
        if id_list:
            subset = df[df["Customer_ID"].astype(str).isin(id_list)].copy()
        else:
            subset = df.iloc[0:0].copy()

    # 需要展示的关键指标（同时用于表格和雷达图）
    metric_config = [
        ("Total_Sales", "消费金额(¥)"),
        ("Total_Orders", "订单数"),
        ("Avg_Order_Value", "用户平均订单价格(¥)"),
        ("Purchase_Frequency", "购买频率"),
        ("Profit_Margin", "利润率"),
        ("Engagement_Score", "互动分"),
        ("Customer_Lifetime", "客户生命周期(天)"),
    ]
    metrics_for_view = [{"key": k, "label": label} for k, label in metric_config]

    # 构造模板渲染用的 rows 列表，每个元素代表一个客户
    rows = []
    for _, r in subset.iterrows():
        metrics_map = {}
        for key, _label in metric_config:
            val = r.get(key)
            if pd.isna(val):
                metrics_map[key] = "-"
            elif isinstance(val, float):
                # 金额与比例保留两位小数，其他保持默认
                if "Sales" in key or "Value" in key or "Margin" in key:
                    metrics_map[key] = f"{val:,.2f}"
                else:
                    metrics_map[key] = f"{val:.2f}"
            else:
                metrics_map[key] = str(val)
        rows.append(
            {
                "customer_id": r.get("Customer_ID"),
                "customer_name": r.get("Customer_Name"),
                "cluster": r.get("Cluster"),
                "cluster_label": r.get("Cluster_Label"),
                "metrics": metrics_map,
            }
        )

    # 构建雷达图（使用各指标百分位 *_Percentile，范围 0-100）
    radar_chart_html = "<p class='empty-hint'>暂无数据用于绘制雷达图。</p>"
    if not subset.empty:
        import plotly.graph_objects as go

        radar_labels = [label for _key, label in metric_config]
        fig = go.Figure()

        # 仅显示前若干个客户，避免图形过于拥挤
        max_traces = 6
        for idx, (_, r) in enumerate(subset.iterrows()):
            if idx >= max_traces:
                break
            vals = []
            for key, _label in metric_config:
                pct_col = f"{key}_Percentile"
                if pct_col in subset.columns:
                    v = r.get(pct_col)
                    v = float(v) if v is not None and not pd.isna(v) else 50.0
                else:
                    v = 50.0
                v = max(0.0, min(100.0, v))
                vals.append(v)
            if not vals:
                continue
            vals_closed = vals + [vals[0]]
            labels_closed = radar_labels + [radar_labels[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals_closed,
                    theta=labels_closed,
                    fill="toself",
                    name=str(r.get("Customer_ID")),
                )
            )

        if fig.data:
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            radar_chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    return render_template_string(
        COMPARE_TEMPLATE,
        ids_text=ids_text,
        rows=rows,
        metrics=metrics_for_view,
        radar_chart_html=radar_chart_html,
    )


def _generate_customer_analysis(r: pd.Series, metrics: list) -> list[dict]:
    """根据聚类、百分位等生成客户分析要点。返回 [{ label, text }] 便于模板渲染。"""
    items = []
    cluster_label = str(r.get("Cluster_Label", r.get("Cluster", "")))

    # 基于聚类的基本定位
    if "高价值" in cluster_label and "流失" not in cluster_label:
        items.append({"label": "群体定位", "text": "该客户属于高价值客户群，消费金额、订单数与购买频率均显著高于平均水平，是核心贡献用户。"})
    elif "中高价值" in cluster_label:
        items.append({"label": "群体定位", "text": "该客户属于中高价值客户群，整体表现优于多数用户，具有较大的提升潜力。"})
    elif "中等价值" in cluster_label:
        items.append({"label": "群体定位", "text": "该客户属于中等价值客户群，处于用户分层中部，可通过精准运营提升转化。"})
    elif "潜在流失" in cluster_label or "流失" in cluster_label:
        items.append({"label": "群体定位", "text": "该客户属于低价值潜在流失客户群，近期活跃度下降，需重点关注和召回。"})
    elif "低价值" in cluster_label:
        items.append({"label": "群体定位", "text": "该客户属于低价值客户群，当前贡献较低，适合通过培育策略挖掘潜力。"})
    else:
        items.append({"label": "群体定位", "text": "该客户已纳入用户分层体系，可根据具体指标制定差异化策略。"})

    # 基于百分位的优势/短板
    strengths = [m for m in metrics if m["percentile"] >= 70]
    weak = [m for m in metrics if m["percentile"] < 40]
    if strengths:
        names = "、".join(s["label"] for s in strengths[:3])
        items.append({"label": "优势维度", "text": f"{names} 处于前 30%，表现突出。"})
    if weak:
        names = "、".join(w["label"] for w in weak[:3])
        items.append({"label": "待提升维度", "text": f"{names} 处于后 40%，可作为重点培育方向。"})

    # 距上次购买天数
    last_days = r.get("Last_Purchase_Days_Ago")
    if last_days is not None and not np.isnan(last_days):
        days = float(last_days)
        if days > 90:
            items.append({"label": "活跃提醒", "text": "距上次购买已超过 90 天，存在明显流失风险。"})
        elif days > 60:
            items.append({"label": "活跃提醒", "text": "近期购买间隔较长，建议通过触达提高复购。"})
        elif days < 14:
            items.append({"label": "活跃表现", "text": "近期购买频繁，用户粘性较好。"})

    # 互动分
    engagement = r.get("Engagement_Score")
    if engagement is not None and not np.isnan(engagement):
        eng_pct = r.get("Engagement_Score_Percentile")
        eng_pct = float(eng_pct) if pd.notna(eng_pct) else 50
        if eng_pct >= 70:
            items.append({"label": "互动表现", "text": "参与度较高，适合推送内容营销和社群运营。"})
        elif eng_pct < 30:
            items.append({"label": "互动表现", "text": "互动较少，可通过问卷、权益等提升参与。"})

    return items


def _generate_marketing_recommendations(r: pd.Series, cluster_label: str) -> list[dict]:
    """根据客户特征生成推荐营销方案。返回 { title, desc, priority } 列表。"""
    recs = []
    cluster = str(cluster_label)

    if "高价值" in cluster and "流失" not in cluster:
        recs.extend([
            {"title": "VIP 专属服务", "desc": "提供专属客服、生日礼遇、优先发货等，强化归属感与忠诚度。", "priority": 1},
            {"title": "复购与交叉销售", "desc": "基于历史偏好推送关联商品或新品，提升用户平均订单价格与复购频次。", "priority": 2},
            {"title": "会员升级激励", "desc": "设计阶梯权益，引导维持或升级会员等级，减少流失。", "priority": 3},
        ])
    elif "中高价值" in cluster:
        recs.extend([
            {"title": "忠诚度计划", "desc": "积分、满减、专属优惠等，促进其向高价值群体转化。", "priority": 1},
            {"title": "品类拓展", "desc": "推荐互补品类或新品试用，提升消费广度与用户平均订单价格。", "priority": 2},
            {"title": "限时促销", "desc": "在关键节点（大促、节日）推送定向优惠，刺激复购。", "priority": 3},
        ])
    elif "中等价值" in cluster:
        recs.extend([
            {"title": "精准促销", "desc": "根据浏览与购买记录，推送高匹配度商品及限时优惠。", "priority": 1},
            {"title": "满额赠礼", "desc": "设置阶梯满减或赠礼，引导提高用户平均订单价格与复购频次。", "priority": 2},
            {"title": "内容与互动", "desc": "通过种草内容、评价有礼等提升互动，增强粘性。", "priority": 3},
        ])
    elif "潜在流失" in cluster or "流失" in cluster:
        recs.extend([
            {"title": "召回与挽留", "desc": "发送专属折扣、满减券或新品推荐，降低回购门槛。", "priority": 1},
            {"title": "客服主动关怀", "desc": "通过电话/短信/邮件询问体验与需求，提供专属解决方案。", "priority": 2},
            {"title": "限时回流礼包", "desc": "限时有效的大额优惠或赠品，营造紧迫感促进回归。", "priority": 3},
        ])
    else:
        # 低价值客户群
        recs.extend([
            {"title": "首购/新人优惠", "desc": "发放首单立减、无门槛券等，降低首次决策成本。", "priority": 1},
            {"title": "拉新与裂变", "desc": "邀请有礼、拼团等活动，利用社交关系扩大转化。", "priority": 2},
            {"title": "培育与种草", "desc": "推送攻略、测评等内容，建立认知后再进行转化。", "priority": 3},
        ])

    return recs


@app.route("/user/random")
def user_random():
    """随机跳转到一位客户的画像报告页。"""
    df = get_profiles_scored()
    if df.empty:
        return "<h1>暂无客户数据</h1><a href='/'>返回</a>", 404
    row = df.sample(n=1).iloc[0]
    cid = row.get("Customer_ID")
    return redirect(f"/user/{cid}", code=302)


@app.route("/user/<customer_id>")
def user_profile(customer_id):
    """客户画像报告页：展示该用户在各指标上的百分位排名及可视化。"""
    df = get_profiles_scored()
    # 从全量画像里找到当前这个 Customer_ID 对应的行
    row = df[df["Customer_ID"].astype(str) == str(customer_id)]
    if row.empty:
        return f"<h1>用户不存在</h1><a href='/'>返回</a>", 404

    r = row.iloc[0]
    cluster_label = r.get("Cluster_Label") or r.get("Cluster")

    # 构建“指标列表”，用于在模板中展示每个指标的数值 + 百分位条
    metric_config = [
        ("Total_Sales", "消费金额", "¥"),
        ("Total_Orders", "订单数", ""),
        ("Avg_Order_Value", "用户平均订单价格", "¥"),
        ("Purchase_Frequency", "购买频率", ""),
        ("Engagement_Score", "互动分", ""),
        ("Customer_Lifetime", "客户生命周期", "天"),
    ]
    metrics = []
    for col, label, unit in metric_config:
        if col not in df.columns:
            continue
        pct_col = f"{col}_Percentile"
        if pct_col not in df.columns:
            continue
        pct = float(r[pct_col]) if pd.notna(r.get(pct_col)) else 50
        val = r.get(col)
        if val is not None and not pd.isna(val) and isinstance(val, (int, float, np.integer, np.floating)):
            if isinstance(val, (float, np.floating)):
                val_str = f"{float(val):,.2f}"
            else:
                val_str = str(int(val))
        else:
            val_str = "-"
        if unit:
            val_str = f"{unit}{val_str}"
        fill_class = "low" if pct < 40 else "mid" if pct < 70 else ""
        marker_pos = max(1.0, min(99.0, pct))
        metrics.append({
            "label": label,
            "value": val_str,
            "percentile": min(100, max(0, pct)),
            "marker_pos": marker_pos,
            "fill_class": fill_class,
        })

    # 雷达图：各指标百分位（同一用户在不同维度的表现）
    radar_labels = [m["label"] for m in metrics]
    radar_values = [m["percentile"] for m in metrics]
    if len(radar_labels) >= 3:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="该客户",
            line={"color": "#2196f3"},
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="各维度百分位雷达图",
            showlegend=False,
            height=400,
        )
        radar_chart_html = fig_radar.to_html(include_plotlyjs="cdn", full_html=False)
    else:
        radar_chart_html = "<p>指标不足，暂不展示雷达图。</p>"

    # 散点图1：消费金额 vs 订单数（灰点=其他用户，红菱形=当前用户）
    x_col, y_col = "Total_Sales", "Total_Orders"
    if x_col in df.columns and y_col in df.columns:
        others = df[df["Customer_ID"].astype(str) != str(customer_id)]
        x_all = others[x_col].fillna(0).values
        y_all = others[y_col].fillna(0).values
        vx, vy = r.get(x_col, 0), r.get(y_col, 0)
        x_cur = 0.0 if (pd.isna(vx) or vx is None) else float(vx)
        y_cur = 0.0 if (pd.isna(vy) or vy is None) else float(vy)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_all, y=y_all, mode="markers", name="其他用户",
                                  marker=dict(size=6, color="rgba(128,128,128,0.5)")))
        fig1.add_trace(go.Scatter(x=[x_cur], y=[y_cur], mode="markers", name="该客户",
                                  marker=dict(size=16, color="#e91e63", symbol="diamond",
                                             line=dict(width=2, color="white"))))
        fig1.update_layout(title="消费金额 vs 订单数", xaxis_title="消费金额", yaxis_title="订单数",
                           height=350, showlegend=True)
        scatter_sales_orders_html = fig1.to_html(include_plotlyjs=False, full_html=False)
    else:
        scatter_sales_orders_html = "<p>缺少数据</p>"

    # 散点图2：用户平均订单价格 vs 购买频率（灰点=其他用户，红菱形=当前用户）
    x_col2, y_col2 = "Avg_Order_Value", "Purchase_Frequency"
    if x_col2 in df.columns and y_col2 in df.columns:
        others2 = df[df["Customer_ID"].astype(str) != str(customer_id)]
        x_all2 = others2[x_col2].fillna(0).values
        y_all2 = others2[y_col2].fillna(0).values
        vx2, vy2 = r.get(x_col2, 0), r.get(y_col2, 0)
        x_cur2 = 0.0 if (pd.isna(vx2) or vx2 is None) else float(vx2)
        y_cur2 = 0.0 if (pd.isna(vy2) or vy2 is None) else float(vy2)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_all2, y=y_all2, mode="markers", name="其他用户",
                                  marker=dict(size=6, color="rgba(128,128,128,0.5)")))
        fig2.add_trace(go.Scatter(x=[x_cur2], y=[y_cur2], mode="markers", name="该客户",
                                  marker=dict(size=16, color="#e91e63", symbol="diamond",
                                             line=dict(width=2, color="white"))))
        fig2.update_layout(title="用户平均订单价格 vs 购买频率", xaxis_title="用户平均订单价格", yaxis_title="购买频率",
                           height=350, showlegend=True)
        scatter_avg_freq_html = fig2.to_html(include_plotlyjs=False, full_html=False)
    else:
        scatter_avg_freq_html = "<p>缺少数据</p>"

    # 订单趋势图：按月份统计（如果有 Month_1_Orders ~ Month_12_Orders）
    orders_trend_html = "<p class='empty-hint'>暂无订单趋势数据。</p>"
    month_cols = [c for c in df.columns if c.startswith("Month_") and c.endswith("_Orders")]
    if month_cols:
        def _month_key(name: str) -> int:
            try:
                return int(name.split("_")[1])
            except Exception:
                return 0

        month_cols = sorted(month_cols, key=_month_key)
        x_labels = [c.split("_")[1] for c in month_cols]
        user_vals = [float(r.get(c) or 0) for c in month_cols]
        avg_vals = [float(df[c].mean() if c in df.columns else 0) for c in month_cols]

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=x_labels,
                y=user_vals,
                name="该客户",
                marker=dict(color="#42a5f5"),
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=x_labels,
                y=avg_vals,
                name="全体平均",
                mode="lines+markers",
                line=dict(color="#ef6c00"),
            )
        )
        fig_trend.update_layout(
            title="近12个月订单数对比",
            xaxis_title="月份",
            yaxis_title="订单数",
            height=360,
            margin=dict(l=40, r=20, t=40, b=60),
        )
        orders_trend_html = fig_trend.to_html(include_plotlyjs=False, full_html=False)

    # 客户分析与营销推荐文案
    analysis_items = _generate_customer_analysis(r, metrics)
    marketing_recs = _generate_marketing_recommendations(r, cluster_label)
    headline_summary = analysis_items[0]["text"] if analysis_items else ""

    # 综合评分 + 运营优先级
    overall_score_val = _compute_overall_score(df, r, str(customer_id))
    priority_label = _compute_priority_label(cluster_label, overall_score_val, r.get("Last_Purchase_Days_Ago"))

    return render_template_string(
        USER_PROFILE_TEMPLATE,
        customer_id=customer_id,
        customer_name=r.get("Customer_Name"),
        cluster=r.get("Cluster"),
        cluster_label=cluster_label,
        favorite_product=r.get("Favorite_Product"),
        overall_score=f"{overall_score_val:.1f} 分",
        priority_label=priority_label,
        metrics=metrics,
        radar_chart_html=radar_chart_html,
        scatter_sales_orders_html=scatter_sales_orders_html,
        scatter_avg_freq_html=scatter_avg_freq_html,
        orders_trend_html=orders_trend_html,
        headline_summary=headline_summary,
        analysis_items=analysis_items,
        marketing_recs=marketing_recs,
        total_sales=_safe(r.get("Total_Sales")),
        total_orders=_safe(r.get("Total_Orders")),
        avg_order_value=_safe(r.get("Avg_Order_Value")),
        customer_lifetime=_safe(r.get("Customer_Lifetime")),
        purchase_frequency=_safe(r.get("Purchase_Frequency")),
        profit_margin=_safe(r.get("Profit_Margin")),
        last_purchase_days=_safe(r.get("Last_Purchase_Days_Ago")),
        engagement_score=_safe(r.get("Engagement_Score")),
    )


@app.route("/")
def index():
    # 读取全量用户画像，并附加“活跃度分数/档位”两列
    df = _add_activity_columns(get_profiles_scored())

    # 读取筛选和排序参数（客户 ID / 群体 / 排序字段 / 升降序 / 分页）
    customer_id = request.args.get("customer_id", "").strip() or None
    cluster = request.args.get("cluster", "").strip() or None
    sort_by = request.args.get("sort_by", "").strip() or "Total_Sales"
    sort_dir = request.args.get("sort_dir", "").strip().lower() or "desc"
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    try:
        per_page = int(request.args.get("per_page", 50))
    except ValueError:
        per_page = 50
    per_page = max(10, min(per_page, 200))
    page = max(1, page)

    # 可选的排序字段配置（用于生成表头的下拉/链接文案）
    sort_options = [
        ("Activity_Score", "活跃度"),
        ("Total_Orders", "订单数"),
        ("Avg_Order_Value", "用户平均订单价格"),
        ("Total_Sales", "消费金额"),
        ("Engagement_Score", "互动分"),
        ("Last_Purchase_Days_Ago", "距上次购买(天)"),
    ]
    allowed_sort_cols = {k for k, _ in sort_options}
    if sort_by not in allowed_sort_cols or sort_by not in df.columns:
        sort_by = "Activity_Score" if "Activity_Score" in df.columns else ("Total_Sales" if "Total_Sales" in df.columns else df.columns[0])
    if sort_dir not in ("asc", "desc"):
        sort_dir = "desc"

    # 聚类分布与概览指标从“全量数据”计算，保证所有分类按钮与统计卡片始终可见
    if "Cluster_Label" in df.columns:
        cluster_key_col = "Cluster_Label"
    else:
        cluster_key_col = "Cluster" if "Cluster" in df.columns else None

    if cluster_key_col is not None:
        cluster_counts_full = (
            df[cluster_key_col].value_counts(dropna=False).sort_index().to_dict()
        )
        cluster_badges = [
            (name, cnt, "/?cluster=" + quote(str(name), safe=""))
            for name, cnt in cluster_counts_full.items()
        ]
    else:
        cluster_counts_full = {}
        cluster_badges = []

    # 概览指标卡片：总销售额 / 总订单数 / 整体客单价 / 高价值客户数 / 潜在流失数
    overall_sales = float(df["Total_Sales"].sum()) if "Total_Sales" in df.columns else 0.0
    overall_orders = float(df["Total_Orders"].sum()) if "Total_Orders" in df.columns else 0.0
    overall_aov = overall_sales / overall_orders if overall_orders > 0 else 0.0
    high_value_count = 0
    churn_count = 0
    if "Cluster_Label" in df.columns:
        labels = df["Cluster_Label"].astype(str)
        high_value_count = int(labels.str.contains("高价值", na=False).sum())
        churn_count = int(labels.str.contains("流失", na=False).sum())

    # 聚类饼图：各客户群体用户占比
    cluster_bar_html = "<p class='empty-hint'>暂无聚类统计数据。</p>"
    if cluster_key_col is not None and cluster_counts_full:
        labels = [str(k) for k in cluster_counts_full.keys()]
        values = [int(v) for v in cluster_counts_full.values()]
        if values and sum(values) > 0:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.35,
                        hoverinfo="label+percent+value",
                    )
                ]
            )
            fig.update_layout(
                height=360,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            cluster_bar_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # 按客户 ID 和群体进行筛选（对筛选后的子集再做排序和分页）
    if customer_id:
        df = df[df["Customer_ID"].astype(str).str.contains(customer_id, case=False)]
    if cluster:
        mask_cluster = df["Cluster"].astype(str) == cluster
        if "Cluster_Label" in df.columns:
            mask_label = df["Cluster_Label"].astype(str) == cluster
            df = df[mask_cluster | mask_label]
        else:
            df = df[mask_cluster]

    # 排序（使用稳定排序 mergesort，方便后续扩展多级排序）
    df = df.sort_values(by=sort_by, ascending=(sort_dir == "asc"), kind="mergesort")

    total_users = len(df)
    total_pages = max(1, (total_users + per_page - 1) // per_page) if total_users else 1
    if page > total_pages:
        page = total_pages
    start = (page - 1) * per_page
    end = start + per_page

    # 只展示前若干列，避免表过宽（首页突出用户基本信息、群体与活跃度）
    display_cols = [
        "Customer_ID",
        "Customer_Name",
        "Cluster_Label",
        "Activity_Score",
        "Total_Orders",
        "Avg_Order_Value",
        "Favorite_Product",
        "Last_Purchase_Days_Ago",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    df_page = df.iloc[start:end]
    df_display = df_page[existing_cols].copy()

    # 用“高/中/低(分数)”展示活跃度，替代原来的消费金额列
    if "Activity_Score" in df_display.columns and "Activity_Level" in df_page.columns:
        df_display["Activity_Score"] = (
            df_page["Activity_Level"].astype(str) + "（" + df_page["Activity_Score"].astype(str) + "）"
        ).values

    # 增加复选框列：用于多选客户并跳转对比页（GET /compare?ids=...）
    raw_ids = df_display["Customer_ID"].astype(str)
    df_display.insert(
        0,
        "选择",
        raw_ids.apply(lambda cid: f'<input type="checkbox" name="ids" value="{cid}" />' if cid and cid != "nan" else ""),
    )
    # 将 Customer_ID 和 Customer_Name 转为可点击链接，跳转至画像报告页
    df_display["Customer_Name"] = df_display.apply(
        lambda row: f'<a href="/user/{row["Customer_ID"]}">{row["Customer_Name"]}</a>'
        if pd.notna(row.get("Customer_Name")) and pd.notna(row.get("Customer_ID"))
        else (str(row.get("Customer_Name", "")) if pd.notna(row.get("Customer_Name")) else ""),
        axis=1,
    )
    df_display["Customer_ID"] = df_display["Customer_ID"].apply(
        lambda x: f'<a href="/user/{x}">{x}</a>' if pd.notna(x) else ""
    )

    # 列名改为中文展示，便于业务同学理解
    rename_map = {
        "选择": "选择",
        "Customer_ID": "客户 ID",
        "Customer_Name": "客户姓名",
        "Cluster_Label": "群体",
        "Cluster": "簇编号",
        "Activity_Score": "活跃度",
        "Total_Sales": "消费金额",
        "Total_Orders": "订单数",
        "Avg_Order_Value": "用户平均订单价格",
        "Favorite_Product": "偏好产品类别",
        "Purchase_Frequency": "购买频率",
        "Profit_Margin": "利润率",
        "Last_Purchase_Days_Ago": "距上次购买(天)",
    }
    df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns}, inplace=True)
    table_html = df_display.to_html(index=False, escape=False)

    # 表头增强：勾选列样式 + 可点击排序表头
    table_html = table_html.replace("<th>选择</th>", '<th class="checkbox-col">选择</th>')

    # 将可排序列的表头改为超链接，点击后根据同一字段升降序切换
    from urllib.parse import quote as _quote_local

    for key, label in sort_options:
        col_label = rename_map.get(key, label)
        header_plain = f"<th>{col_label}</th>"
        if header_plain not in table_html:
            continue
        # 点击当前列时切换排序方向，否则默认降序
        if sort_by == key:
            arrow = "▲" if sort_dir == "asc" else "▼"
            next_dir = "desc" if sort_dir == "asc" else "asc"
        else:
            arrow = ""
            next_dir = "desc"
        href = (
            "/?page=1"
            f"&per_page={per_page}"
            f"&customer_id={_quote_local(customer_id or '')}"
            f"&cluster={_quote_local(cluster or '')}"
            f"&sort_by={key}"
            f"&sort_dir={next_dir}"
        )
        link_text = f"{col_label} {arrow}" if arrow else col_label
        header_link = f'<th><a href="{href}">{link_text}</a></th>'
        table_html = table_html.replace(header_plain, header_link, 1)

    return render_template_string(
        INDEX_TEMPLATE,
        total_users=total_users,
        total_pages=total_pages,
        page=page,
        per_page=per_page,
        cluster_badges=cluster_badges,
        table_html=table_html,
        customer_id=customer_id,
        cluster=cluster,
        sort_by=sort_by,
        sort_dir=sort_dir,
        sort_options=sort_options,
        overall_sales=overall_sales,
        overall_orders=overall_orders,
        overall_aov=overall_aov,
        high_value_count=high_value_count,
        churn_count=churn_count,
        cluster_bar_html=cluster_bar_html,
    )


# === JSON API（供 Vue 等前端调用） ===


@app.route("/api/cluster-summary")
def api_cluster_summary():
    """返回各聚类群体的汇总统计。"""
    df = get_profiles_scored()
    cluster_col = "Cluster_Label" if "Cluster_Label" in df.columns else "Cluster"
    if cluster_col not in df.columns:
        return jsonify({"error": "无聚类列"}), 400

    numeric_cols = [
        "Total_Sales",
        "Total_Orders",
        "Avg_Order_Value",
        "Customer_Lifetime",
        "Purchase_Frequency",
        "Profit_Margin",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    summary = []
    for label, grp in df.groupby(cluster_col):
        rec = {
            "cluster": str(label),
            "count": int(len(grp)),
            "pct": round(len(grp) / len(df) * 100, 1),
        }
        for col in numeric_cols:
            rec[f"avg_{col}"] = round(float(grp[col].mean()), 2)
        summary.append(rec)
    return jsonify({"total_users": len(df), "clusters": summary})


@app.route("/api/users")
def api_users():
    """分页查询用户列表。支持 customer_id、cluster 过滤。"""
    df = get_profiles_scored()
    customer_id = request.args.get("customer_id", "").strip() or None
    cluster = request.args.get("cluster", "").strip() or None
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(10, int(request.args.get("per_page", 20))))

    if customer_id:
        df = df[df["Customer_ID"].astype(str).str.contains(customer_id, case=False)]
    if cluster:
        mask = df["Cluster"].astype(str) == cluster
        if "Cluster_Label" in df.columns:
            mask = mask | (df["Cluster_Label"].astype(str) == cluster)
        df = df[mask]

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    display_cols = [
        "Customer_ID",
        "Customer_Name",
        "Cluster_Label",
        "Cluster",
        "Total_Sales",
        "Total_Orders",
        "Avg_Order_Value",
        "Purchase_Frequency",
    ]
    display_cols = [c for c in display_cols if c in page_df.columns]
    rows = page_df[display_cols].fillna("").to_dict(orient="records")
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer, np.floating)):
                r[k] = float(v) if np.issubdtype(type(v), np.floating) else int(v)
    return jsonify(
        {"total": total, "page": page, "per_page": per_page, "data": rows}
    )


@app.route("/api/user/<customer_id>")
def api_user_detail(customer_id):
    """返回单个用户的详细画像。"""
    df = get_profiles_scored()
    row = df[df["Customer_ID"].astype(str) == str(customer_id)]
    if row.empty:
        return jsonify({"error": "用户不存在"}), 404
    rec = row.iloc[0].to_dict()
    for k, v in list(rec.items()):
        if pd.isna(v):
            rec[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            rec[k] = float(v) if np.issubdtype(type(v), np.floating) else int(v)
    return jsonify(rec)


if __name__ == "__main__":
    # debug=True 方便开发时自动重载
    app.run(host="127.0.0.1", port=5000, debug=True)

