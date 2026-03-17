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
import time
from urllib.parse import quote

from flask import Flask, jsonify, redirect, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go


app = Flask(__name__)

DATA_CSV_PATH = "customer_features_rfmbc.csv"
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
            # 兼容字符串形式的数字（如 "58" / "58.0"）
            if isinstance(v, str):
                s = v.strip()
                if s == "":
                    return int(default)
                try:
                    return int(float(s))
                except Exception:
                    return int(default)
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return int(default)

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


# === 简单缓存：避免每个请求都全量重算 ===
# 说明：
# - 首页/详情页/对比页都会调用 get_profiles_scored()，如果每次请求都重新 rank/percentile，会明显变慢。
# - 这里做一个轻量的内存缓存（进程内），对开发/演示很有用。
_PROFILES_CACHE: pd.DataFrame | None = None
_PROFILES_CACHE_KEY: tuple | None = None
_PROFILES_CACHE_AT: float = 0.0


def _profiles_cache_key() -> tuple:
    """生成缓存 key：CSV 模式跟随文件 mtime；MySQL 模式使用连接配置 + 固定 key。"""
    if USE_MYSQL:
        # MySQL 的数据变化无法可靠感知；此处只把配置纳入 key，并依赖 TTL 自动刷新
        return (
            "mysql",
            MYSQL_CONFIG.get("host"),
            MYSQL_CONFIG.get("port"),
            MYSQL_CONFIG.get("user"),
            MYSQL_CONFIG.get("database"),
            MYSQL_TABLE,
        )
    try:
        st = os.stat(DATA_CSV_PATH)
        return ("csv", DATA_CSV_PATH, int(st.st_mtime), int(st.st_size))
    except OSError:
        # 文件不存在/不可访问时也要有 key，避免异常导致缓存逻辑崩
        return ("csv", DATA_CSV_PATH, 0, 0)


def get_profiles_scored_cached(ttl_sec: float = 5.0) -> pd.DataFrame:
    """带 TTL 的缓存版本：尽量复用上一份已计算的画像表。"""
    global _PROFILES_CACHE, _PROFILES_CACHE_KEY, _PROFILES_CACHE_AT
    key = _profiles_cache_key()
    now = time.time()
    if (
        _PROFILES_CACHE is not None
        and _PROFILES_CACHE_KEY == key
        and (now - _PROFILES_CACHE_AT) <= float(ttl_sec)
    ):
        return _PROFILES_CACHE
    df = get_profiles_scored()
    _PROFILES_CACHE = df
    _PROFILES_CACHE_KEY = key
    _PROFILES_CACHE_AT = now
    return df


# === Flask 路由 ===

def _pick_biz_name_by_segment_rank(
    seg_sales: float | None,
    seg_freq: float | None,
    seg_churn90: float | None,
    all_sales: list[float],
    all_freq: list[float],
    all_churn90: list[float],
) -> tuple[str, str]:
    """
    基于“群体之间的相对排名”生成业务命名，避免用整体均值倍数导致命名趋同。
    返回：(业务命名, 命名依据文本)
    """
    def _percentile_rank(v: float | None, arr: list[float], higher_better: bool) -> float | None:
        """
        返回 v 在 arr 中的分位（0~1），越大代表“越靠前/越好”。
        higher_better=True：值越大越好；False：值越小越好（如流失风险）。
        """
        if v is None or pd.isna(v) or not arr:
            return None
        vals = [float(x) for x in arr if x is not None and not pd.isna(x)]
        if not vals:
            return None
        n = len(vals)
        if n == 1:
            return 1.0
        if higher_better:
            # 统计 <= v 的比例作为分位
            r = sum(1 for x in vals if x <= float(v)) / n
        else:
            # 值越小越好：统计 >= v 的比例作为“好分位”
            r = sum(1 for x in vals if x >= float(v)) / n
        return max(0.0, min(1.0, float(r)))

    # 销售额/频率：越高越好；churn90：越低越好（因此 higher_better=False）
    sales_pos = _percentile_rank(seg_sales, all_sales, higher_better=True)
    freq_pos = _percentile_rank(seg_freq, all_freq, higher_better=True)
    churn_pos = _percentile_rank(seg_churn90, all_churn90, higher_better=False)

    # 用更宽松阈值拉开 5 类区分度（群体数量通常不大）
    def _hi(p): return p is not None and p >= 0.70
    def _lo(p): return p is not None and p <= 0.30

    if _hi(sales_pos) and _hi(churn_pos):
        name = "高价值高忠诚用户"
    elif _hi(sales_pos) and _lo(churn_pos):
        name = "高价值需挽留用户"
    elif _lo(sales_pos) and _lo(churn_pos):
        name = "低价值高流失风险用户"
    elif _hi(freq_pos) and _lo(sales_pos):
        name = "中低价值潜力用户"
    elif _hi(freq_pos):
        name = "中价值活跃增长用户"
    else:
        name = "低价值低活跃用户"

    basis_parts = []
    if sales_pos is not None:
        basis_parts.append(f"销售额分位 {sales_pos:.0%}")
    if freq_pos is not None:
        basis_parts.append(f"频率分位 {freq_pos:.0%}")
    if churn_pos is not None:
        basis_parts.append(f"低流失分位 {churn_pos:.0%}")
    basis = "；".join(basis_parts) if basis_parts else "基于群体相对排名判定"
    return name, basis

def _prepare_X_for_clustering(df_raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """为聚类实验准备特征矩阵：选择可用列、填充缺失值并做 Min-Max 归一化到 [0,1]。"""
    from sklearn.preprocessing import MinMaxScaler

    feature_aliases = {
        "Total_Sales": ["Total_Sales"],
        "Total_Orders": ["Total_Orders"],
        "Avg_Order_Value": ["Avg_Order_Value"],
        "Customer_Lifetime": ["Customer_Lifetime_Days", "Customer_Lifetime"],
        "Purchase_Frequency": ["Purchase_Frequency"],
        "Avg_Browsing_Time": ["Avg_Browsing_Time"],
        "Profit_Margin": ["Profit_Margin"],
        "Days_Since_Last_Purchase": ["Days_Since_Last_Purchase", "Last_Purchase_Days_Ago"],
        "Engagement_Score": ["Total_Engagement_Score", "Engagement_Score"],
        "Unique_Products": ["Unique_Products_Purchased", "Unique_Products_Bought"],
    }
    cols = list(df_raw.columns)
    available: list[str] = []
    for _canon, aliases in feature_aliases.items():
        for a in aliases:
            if a in cols:
                available.append(a)
                break

    if not available:
        raise RuntimeError("未找到可用于聚类的特征列（请检查 CSV 列名）")

    X = df_raw[available].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, available


@app.route("/algo-compare")
def algo_compare():
    """论文/答辩用：K-Means 与 DBSCAN 对比页（轮廓系数、CH、耗时、簇数、噪声比例）。"""
    error = None
    try:
        # 聚类实验更适合用原始 CSV（避免 MySQL 全 TEXT 带来的类型问题）
        df_raw = pd.read_csv(DATA_CSV_PATH)
        X, features = _prepare_X_for_clustering(df_raw)

        import time as _time
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        # 1) K-Means：遍历 K（用于手肘法 + 轮廓系数曲线）
        k_range = [3, 4, 5, 6]
        kmeans_rows = []
        best_k = None
        best_sil = float("-inf")
        k_list: list[int] = []
        sse_list: list[float] = []
        sil_list: list[float] = []
        for k in k_range:
            t0 = _time.perf_counter()
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            elapsed = _time.perf_counter() - t0

            sse = float(km.inertia_)
            n_clusters = len(np.unique(labels[labels >= 0]))
            if 1 < n_clusters < len(X):
                sil = float(silhouette_score(X, labels))
                ch = float(calinski_harabasz_score(X, labels))
            else:
                sil = float("nan")
                ch = float("nan")

            k_list.append(int(k))
            sse_list.append(float(sse))
            sil_list.append(float(sil) if not np.isnan(sil) else float("nan"))

            if not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_k = k

            kmeans_rows.append(
                {
                    "k": k,
                    "sse": f"{sse:,.0f}",
                    "sil": f"{sil:.4f}" if not np.isnan(sil) else "N/A",
                    "ch": f"{ch:.2f}" if not np.isnan(ch) else "N/A",
                    "time": f"{elapsed:.3f}",
                }
            )
        if best_k is None:
            best_k = 5
            best_sil = float("nan")

        # K 值选择可视化：SSE 手肘图 + Silhouette 曲线
        elbow_html = "<p class='sub'>暂无数据</p>"
        sil_html = "<p class='sub'>暂无数据</p>"
        if k_list and sse_list:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(
                go.Scatter(
                    x=k_list,
                    y=sse_list,
                    mode="lines+markers",
                    line=dict(color="#1976d2", width=2),
                    marker=dict(size=7, line=dict(width=1, color="#ffffff")),
                    name="SSE",
                )
            )
            fig_elbow.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(title="K（簇数）", dtick=1),
                yaxis=dict(title="SSE（簇内误差平方和）"),
                showlegend=False,
            )
            elbow_html = fig_elbow.to_html(include_plotlyjs="cdn", full_html=False)

        if k_list and sil_list and any((not np.isnan(v)) for v in sil_list):
            fig_sil = go.Figure()
            fig_sil.add_trace(
                go.Scatter(
                    x=k_list,
                    y=[None if np.isnan(v) else float(v) for v in sil_list],
                    mode="lines+markers",
                    line=dict(color="#2e7d32", width=2),
                    marker=dict(size=7, line=dict(width=1, color="#ffffff")),
                    name="Silhouette",
                )
            )
            # 标注推荐 K
            if best_k is not None and not np.isnan(best_sil):
                fig_sil.add_trace(
                    go.Scatter(
                        x=[int(best_k)],
                        y=[float(best_sil)],
                        mode="markers+text",
                        text=[f"推荐 K={int(best_k)}"],
                        textposition="top center",
                        marker=dict(size=10, color="#e91e63", line=dict(width=1, color="#ffffff")),
                        showlegend=False,
                    )
                )
            fig_sil.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(title="K（簇数）", dtick=1),
                yaxis=dict(title="Silhouette（轮廓系数）"),
                showlegend=False,
            )
            sil_html = fig_sil.to_html(include_plotlyjs=False, full_html=False)

        # 2) DBSCAN：小网格搜索
        eps_range = [0.3, 0.4, 0.5, 0.6, 0.7]
        min_samples_range = [3, 5, 7]
        db_rows = []
        best = None

        def _score_db(rec: dict) -> tuple:
            # 排序规则：满足约束优先，其次 silhouette 越高越好
            ok = (rec["n_clusters"] >= 2) and (rec["noise_ratio_pct"] < 10) and (not np.isnan(rec["silhouette"]))
            sil = rec["silhouette"] if not np.isnan(rec["silhouette"]) else -999.0
            return (1 if ok else 0, sil)

        for eps in eps_range:
            for ms in min_samples_range:
                t0 = _time.perf_counter()
                db = DBSCAN(eps=eps, min_samples=ms)
                labels = db.fit_predict(X)
                elapsed = _time.perf_counter() - t0

                n_noise = int((labels == -1).sum())
                noise_ratio = float(n_noise) / float(len(labels)) * 100.0
                n_clusters = int(len(np.unique(labels)) - (1 if -1 in labels else 0))

                if n_clusters > 1:
                    mask = labels >= 0
                    if int(mask.sum()) > n_clusters:
                        sil = float(silhouette_score(X[mask], labels[mask]))
                        ch = float(calinski_harabasz_score(X[mask], labels[mask]))
                    else:
                        sil = float("nan")
                        ch = float("nan")
                else:
                    sil = float("nan")
                    ch = float("nan")

                rec = {
                    "eps": eps,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "noise_ratio_pct": noise_ratio,
                    "silhouette": sil,
                    "ch": ch,
                    "time_sec": elapsed,
                }
                if best is None or _score_db(rec) > _score_db(best):
                    best = rec

                db_rows.append(
                    {
                        "eps": f"{eps:.1f}",
                        "min_samples": ms,
                        "n_clusters": n_clusters,
                        "noise_ratio": f"{noise_ratio:.1f}",
                        "sil": f"{sil:.4f}" if not np.isnan(sil) else "N/A",
                        "ch": f"{ch:.2f}" if not np.isnan(ch) else "N/A",
                        "time": f"{elapsed:.3f}",
                    }
                )

        if best is None:
            best = {"eps": 0.5, "min_samples": 5, "n_clusters": 0, "noise_ratio_pct": 0.0, "silhouette": float("nan")}

        # 3) 最终对比：用推荐参数各跑一次
        t0 = _time.perf_counter()
        km = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
        km_labels = km.fit_predict(X)
        km_time = _time.perf_counter() - t0
        km_sil = float(silhouette_score(X, km_labels))
        km_ch = float(calinski_harabasz_score(X, km_labels))

        t0 = _time.perf_counter()
        db = DBSCAN(eps=float(best["eps"]), min_samples=int(best["min_samples"]))
        db_labels = db.fit_predict(X)
        db_time = _time.perf_counter() - t0
        mask = db_labels >= 0
        db_n_clusters = int(len(np.unique(db_labels[mask])))
        if db_n_clusters >= 2 and int(mask.sum()) > db_n_clusters:
            db_sil = float(silhouette_score(X[mask], db_labels[mask]))
            db_ch = float(calinski_harabasz_score(X[mask], db_labels[mask]))
        else:
            db_sil = float("nan")
            db_ch = float("nan")
        db_noise = float((db_labels == -1).sum()) / float(len(db_labels)) * 100.0

        compare = {
            "kmeans": {
                "n_clusters": int(best_k),
                "sil": f"{km_sil:.4f}",
                "ch": f"{km_ch:.2f}",
                "time": f"{km_time:.3f}",
            },
            "dbscan": {
                "n_clusters": int(db_n_clusters),
                "noise_ratio": f"{db_noise:.1f}",
                "sil": f"{db_sil:.4f}" if not np.isnan(db_sil) else "N/A",
                "ch": f"{db_ch:.2f}" if not np.isnan(db_ch) else "N/A",
                "time": f"{db_time:.3f}",
            },
        }

        db_best_view = {
            "eps": f"{float(best['eps']):.1f}",
            "min_samples": int(best["min_samples"]),
            "n_clusters": int(best.get("n_clusters", 0)),
            "noise_ratio": f"{float(best.get('noise_ratio_pct', 0.0)):.1f}",
            "sil": f"{float(best.get('silhouette', float('nan'))):.4f}" if not np.isnan(float(best.get("silhouette", float("nan")))) else "N/A",
        }

        # 论文可引用总结：自动拼接关键结论（含数值）
        def _nz(x: str) -> str:
            return x if x and x != "N/A" else "N/A"

        paper_summary = (
            "在同一数据集与相同特征归一化设置下，对 K-Means 与 DBSCAN 进行聚类对比实验。"
            f"K-Means 通过遍历 K∈{k_range} 并计算 SSE（手肘法）与轮廓系数（Silhouette）确定最佳簇数，"
            f"本次实验推荐 K={int(best_k)}（Silhouette={_nz(compare['kmeans']['sil'])}，CH={_nz(compare['kmeans']['ch'])}，耗时={compare['kmeans']['time']}s）。"
            f"DBSCAN 通过网格搜索 eps∈{eps_range}、min_samples∈{min_samples_range}，优先满足簇数≥2 且噪声比例<10%，并尽量提升轮廓系数，"
            f"本次实验推荐 eps={db_best_view['eps']}、min_samples={db_best_view['min_samples']}（簇数={db_best_view['n_clusters']}，噪声={db_best_view['noise_ratio']}%，Silhouette={db_best_view['sil']}）。"
            "最终对比显示："
            f"K-Means（簇数={compare['kmeans']['n_clusters']}，Silhouette={_nz(compare['kmeans']['sil'])}，CH={_nz(compare['kmeans']['ch'])}）"
            f"与 DBSCAN（簇数={compare['dbscan']['n_clusters']}，噪声={compare['dbscan']['noise_ratio']}%，Silhouette={_nz(compare['dbscan']['sil'])}，CH={_nz(compare['dbscan']['ch'])}）在质量与可用性上各有取舍。"
            "结合业务解释性需求（需要固定、稳定的用户分层标签），K-Means 更适合作为最终用户分群方法；"
            "DBSCAN 更适合用于识别离群/噪声样本或作为补充分析。"
        )

        return render_template(
            "algo_compare.html",
            error=None,
            data_source=DATA_CSV_PATH,
            n_samples=int(X.shape[0]),
            features=features,
            elbow_html=elbow_html,
            sil_html=sil_html,
            kmeans_table=kmeans_rows,
            kmeans_best_k=int(best_k),
            kmeans_best_sil=f"{best_sil:.4f}" if not np.isnan(best_sil) else "N/A",
            dbscan_table=db_rows,
            db_best=db_best_view,
            compare=compare,
            paper_summary=paper_summary,
        )
    except Exception as e:
        error = str(e)
        return render_template("algo_compare.html", error=error)


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
    df = get_profiles_scored_cached()

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

    return render_template(
        "compare.html",
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
    df = get_profiles_scored_cached()
    if df.empty:
        return "<h1>暂无客户数据</h1><a href='/'>返回</a>", 404
    row = df.sample(n=1).iloc[0]
    cid = row.get("Customer_ID")
    return redirect(f"/user/{cid}", code=302)


@app.route("/user/<customer_id>")
def user_profile(customer_id):
    """客户画像报告页：展示该用户在各指标上的百分位排名及可视化。"""
    df = get_profiles_scored_cached()
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

    return render_template(
        "user_profile.html",
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
    df = _add_activity_columns(get_profiles_scored_cached())
    df_full = df.copy()

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

    # 群体核心特征概览（按筛选显示）：仅当筛选了某个群体时展示该群体概览
    selected_segment = None
    if cluster and cluster_key_col is not None and cluster_key_col in df_full.columns and not df_full.empty:
        def _mean_num(frame: pd.DataFrame, col: str) -> float:
            return float(pd.to_numeric(frame[col], errors="coerce").mean())

        overall_sales = _mean_num(df_full, "Total_Sales") if "Total_Sales" in df_full.columns else np.nan
        overall_orders = _mean_num(df_full, "Total_Orders") if "Total_Orders" in df_full.columns else np.nan
        overall_freq = _mean_num(df_full, "Purchase_Frequency") if "Purchase_Frequency" in df_full.columns else np.nan

        # 用“群体之间的相对排名”做业务命名（避免整体均值被极端值拉偏导致命名趋同）
        grp_full = df_full.groupby(cluster_key_col, dropna=False)
        all_sales = [
            _mean_num(_g, "Total_Sales") for _seg, _g in grp_full
            if "Total_Sales" in _g.columns
        ]
        all_freq = [
            _mean_num(_g, "Purchase_Frequency") for _seg, _g in grp_full
            if "Purchase_Frequency" in _g.columns
        ]
        all_churn90 = [
            float((pd.to_numeric(_g["Last_Purchase_Days_Ago"], errors="coerce") > 90).mean()) * 100.0
            for _seg, _g in grp_full
            if "Last_Purchase_Days_Ago" in _g.columns
        ]

        def _safe_ratio(a: float, b: float) -> float | None:
            if b is None or pd.isna(b) or b == 0 or a is None or pd.isna(a):
                return None
            return float(a) / float(b)

        def _pct(series: pd.Series, thr: float) -> float:
            s = pd.to_numeric(series, errors="coerce")
            if s.dropna().empty:
                return 0.0
            return round(float((s > thr).mean()) * 100.0, 1)

        # 命名逻辑改为：由群体之间的相对排名决定（更稳定、更容易拉开区分度）

        # 只计算当前筛选群体的概览（不再一直展示全部群体）
        mask_label = df_full[cluster_key_col].astype(str) == str(cluster)
        mask_cluster = df_full["Cluster"].astype(str) == str(cluster) if "Cluster" in df_full.columns else False
        gseg = df_full[mask_label | mask_cluster]
        if not gseg.empty:
            name = str(cluster)
            cnt = int(len(gseg))
            pct = round(cnt / len(df_full) * 100.0, 1) if len(df_full) else 0.0
            href = "/?cluster=" + quote(name, safe="")

            sales_avg = _mean_num(gseg, "Total_Sales") if "Total_Sales" in gseg.columns else np.nan
            orders_avg = _mean_num(gseg, "Total_Orders") if "Total_Orders" in gseg.columns else np.nan
            freq_avg = _mean_num(gseg, "Purchase_Frequency") if "Purchase_Frequency" in gseg.columns else np.nan
            churn60 = _pct(gseg["Last_Purchase_Days_Ago"], 60) if "Last_Purchase_Days_Ago" in gseg.columns else 0.0
            churn90 = _pct(gseg["Last_Purchase_Days_Ago"], 90) if "Last_Purchase_Days_Ago" in gseg.columns else 0.0

            sales_r = _safe_ratio(sales_avg, overall_sales)
            orders_r = _safe_ratio(orders_avg, overall_orders)
            freq_r = _safe_ratio(freq_avg, overall_freq)
            biz, biz_basis = _pick_biz_name_by_segment_rank(
                seg_sales=None if pd.isna(sales_avg) else float(sales_avg),
                seg_freq=None if pd.isna(freq_avg) else float(freq_avg),
                seg_churn90=float(churn90),
                all_sales=all_sales,
                all_freq=all_freq,
                all_churn90=all_churn90,
            )

            diff_parts = []
            if sales_r is not None:
                diff_parts.append(f"销售额≈整体{sales_r:.2f}×")
            if orders_r is not None:
                diff_parts.append(f"订单数≈整体{orders_r:.2f}×")
            if freq_r is not None:
                diff_parts.append(f"频率≈整体{freq_r:.2f}×")
            diff_parts.append(f">90天未购{churn90:.1f}%")

            selected_segment = {
                "name": name,
                "href": href,
                "biz_name": biz,
                "biz_basis": biz_basis,
                "count": cnt,
                "pct": pct,
                "sales_avg": "-" if pd.isna(sales_avg) else f"{sales_avg:,.2f}",
                "orders_avg": "-" if pd.isna(orders_avg) else f"{orders_avg:,.2f}",
                "freq_avg": "-" if pd.isna(freq_avg) else f"{freq_avg:,.4f}",
                "churn60": f"{churn60:.1f}%",
                "churn90": f"{churn90:.1f}%",
                "diff_text": "；".join(diff_parts),
            }

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

    return render_template(
        "index.html",
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
        selected_segment=selected_segment,
    )


# === JSON API（供 Vue 等前端调用） ===


@app.route("/api/cluster-summary")
def api_cluster_summary():
    """返回各聚类群体的汇总统计。"""
    df = get_profiles_scored_cached()
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
    df = get_profiles_scored_cached()
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
    df = get_profiles_scored_cached()
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


@app.route("/segment-overview")
def segment_overview():
    """群体总览页：输出各群体关键指标对比 + 雷达图/热力图，用于答辩展示与运营解读。"""
    # 1) 读取全量画像（带百分位/排名），并补齐活跃度列，便于群体对比
    df = _add_activity_columns(get_profiles_scored_cached())
    if df.empty:
        return "<h1>暂无客户数据</h1><a href='/'>返回</a>", 404

    # 2) 决定群体列（优先用 Cluster_Label，否则退回 Cluster）
    seg_col = "Cluster_Label" if "Cluster_Label" in df.columns else ("Cluster" if "Cluster" in df.columns else None)
    if seg_col is None:
        return "<h1>数据中缺少群体列（Cluster_Label/Cluster）</h1><a href='/'>返回</a>", 400

    # 3) 总览卡片
    total_users = int(len(df))
    labels = df[seg_col].astype(str)
    high_value_count = int(labels.str.contains("高价值", na=False).sum()) if seg_col == "Cluster_Label" else 0
    churn_count = int(labels.str.contains("流失", na=False).sum()) if seg_col == "Cluster_Label" else 0

    # 4) 群体对比指标（表格与图表共用）
    metrics = [
        ("Total_Sales", "消费金额"),
        ("Total_Orders", "订单数"),
        ("Avg_Order_Value", "用户平均订单价格"),
        ("Purchase_Frequency", "购买频率"),
        ("Engagement_Score", "互动分"),
        ("Customer_Lifetime", "客户生命周期"),
    ]
    pct_metrics = [(f"{k}_Percentile", label) for k, label in metrics if f"{k}_Percentile" in df.columns]
    # “最近购买”是越小越好，这里直接用 Last_Purchase_Days_Ago 的反向标准化做群体图展示
    has_last = "Last_Purchase_Days_Ago" in df.columns

    # 5) 群体统计表（均值/中位数 + 核心特征对比 + 业务命名 + 样本客户）
    segments = []
    grp = df.groupby(seg_col, dropna=False)

    # --- 核心特征对比：用于论文中的“差异解释/合理性验证” ---
    def _mean_num(frame: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(frame[col], errors="coerce").mean())

    overall_sales = _mean_num(df, "Total_Sales") if "Total_Sales" in df.columns else np.nan
    overall_orders = _mean_num(df, "Total_Orders") if "Total_Orders" in df.columns else np.nan
    overall_freq = _mean_num(df, "Purchase_Frequency") if "Purchase_Frequency" in df.columns else np.nan

    def _safe_ratio(a: float, b: float) -> float | None:
        if b is None or pd.isna(b) or b == 0 or a is None or pd.isna(a):
            return None
        return float(a) / float(b)

    def _pct(series: pd.Series, thr: float) -> float:
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().empty:
            return 0.0
        return round(float((s > thr).mean()) * 100.0, 1)

    # 业务命名不再使用“整体均值倍数阈值”（容易因为极端值导致命名趋同），改为“群体之间的分位/排名”

    seg_core = {}
    for _seg_name, _g in grp:
        key = str(_seg_name)
        s_avg = _mean_num(_g, "Total_Sales") if "Total_Sales" in _g.columns else np.nan
        f_avg = _mean_num(_g, "Purchase_Frequency") if "Purchase_Frequency" in _g.columns else np.nan
        c90 = _pct(_g["Last_Purchase_Days_Ago"], 90) if "Last_Purchase_Days_Ago" in _g.columns else 0.0
        seg_core[key] = {"sales": s_avg, "freq": f_avg, "churn90": c90}

    all_sales = [v["sales"] for v in seg_core.values() if not pd.isna(v["sales"])]
    all_freq = [v["freq"] for v in seg_core.values() if not pd.isna(v["freq"])]
    all_churn90 = [v["churn90"] for v in seg_core.values()]
    for seg_name, g in grp:
        name = str(seg_name)
        cnt = int(len(g))
        pct = round(cnt / total_users * 100, 1) if total_users else 0.0
        href = "/?cluster=" + quote(name, safe="")

        def _mean(col: str) -> str:
            if col not in g.columns:
                return "-"
            v = float(pd.to_numeric(g[col], errors="coerce").mean())
            return "-" if pd.isna(v) else f"{v:,.2f}"

        def _median(col: str) -> str:
            if col not in g.columns:
                return "-"
            v = float(pd.to_numeric(g[col], errors="coerce").median())
            return "-" if pd.isna(v) else f"{v:,.1f}"

        # 找“优势维度”：取群体百分位均值最高的前 2 个
        strengths = "-"
        if pct_metrics:
            means = []
            for col, label in pct_metrics:
                v = float(pd.to_numeric(g[col], errors="coerce").mean())
                if pd.isna(v):
                    continue
                means.append((v, label))
            means.sort(key=lambda x: x[0], reverse=True)
            if means:
                strengths = "、".join([m[1] for m in means[:2]])

        # 样本客户：优先按活跃度分数，其次按消费金额
        order_cols = []
        if "Activity_Score" in g.columns:
            order_cols.append("Activity_Score")
        if "Total_Sales" in g.columns:
            order_cols.append("Total_Sales")
        if order_cols:
            sample_df = g.sort_values(by=order_cols, ascending=False).head(3)
        else:
            sample_df = g.head(3)

        samples = [
            {"id": str(r.get("Customer_ID")), "name": str(r.get("Customer_Name") or "-")}
            for _, r in sample_df.iterrows()
            if pd.notna(r.get("Customer_ID"))
        ]

        # 核心特征均值与风险：用于“聚类核心区别”的量化说明
        sales_avg_num = _mean_num(g, "Total_Sales") if "Total_Sales" in g.columns else np.nan
        orders_avg_num = _mean_num(g, "Total_Orders") if "Total_Orders" in g.columns else np.nan
        freq_avg_num = _mean_num(g, "Purchase_Frequency") if "Purchase_Frequency" in g.columns else np.nan

        churn60 = _pct(g["Last_Purchase_Days_Ago"], 60) if "Last_Purchase_Days_Ago" in g.columns else 0.0
        churn90 = _pct(g["Last_Purchase_Days_Ago"], 90) if "Last_Purchase_Days_Ago" in g.columns else 0.0

        sales_ratio = _safe_ratio(sales_avg_num, overall_sales)
        orders_ratio = _safe_ratio(orders_avg_num, overall_orders)
        freq_ratio = _safe_ratio(freq_avg_num, overall_freq)

        biz_name, biz_basis = _pick_biz_name_by_segment_rank(
            seg_sales=None if pd.isna(sales_avg_num) else float(sales_avg_num),
            seg_freq=None if pd.isna(freq_avg_num) else float(freq_avg_num),
            seg_churn90=float(churn90),
            all_sales=all_sales,
            all_freq=all_freq,
            all_churn90=all_churn90,
        )

        diff_bits = []
        if sales_ratio is not None:
            diff_bits.append(f"销售额≈整体{sales_ratio:.2f}×")
        if orders_ratio is not None:
            diff_bits.append(f"订单数≈整体{orders_ratio:.2f}×")
        if freq_ratio is not None:
            diff_bits.append(f"频率≈整体{freq_ratio:.2f}×")
        diff_bits.append(f">90天未购{churn90:.1f}%")
        diff_text = "；".join(diff_bits)

        segments.append(
            {
                "name": name,
                "href": href,
                "count": cnt,
                "pct": pct,
                "activity_avg": _mean("Activity_Score"),
                "sales_avg": _mean("Total_Sales"),
                "orders_avg": _mean("Total_Orders"),
                "aov_avg": _mean("Avg_Order_Value"),
                "last_days_median": _median("Last_Purchase_Days_Ago"),
                "strengths": strengths,
                "freq_avg": _mean("Purchase_Frequency"),
                "churn60_pct": f"{churn60:.1f}%",
                "churn90_pct": f"{churn90:.1f}%",
                "biz_name": biz_name,
                "biz_basis": biz_basis,
                "diff_text": diff_text,
                "samples": samples,
            }
        )

    # 排序：按人数占比从高到低
    segments.sort(key=lambda x: x["count"], reverse=True)

    # 5.1) 合理性验证：关键指标在不同群体之间是否存在“量级差异”
    validation_notes: list[str] = []
    key_labels = [
        ("Total_Sales", "平均销售额"),
        ("Total_Orders", "平均订单数"),
        ("Purchase_Frequency", "购买频率"),
        ("Last_Purchase_Days_Ago", "距上次购买(天)"),
    ]
    for col, label in key_labels:
        if col not in df.columns:
            continue
        vals = []
        for seg_name, g in grp:
            v = float(pd.to_numeric(g[col], errors="coerce").mean())
            if pd.isna(v):
                continue
            vals.append((str(seg_name), v))
        if len(vals) < 2:
            continue
        vals.sort(key=lambda x: x[1])
        min_name, min_v = vals[0]
        max_name, max_v = vals[-1]
        if min_v == 0:
            continue
        validation_notes.append(
            f"{label}：最大群体「{max_name}」({max_v:,.2f}) vs 最小群体「{min_name}」({min_v:,.2f})，差异约 {max_v/min_v:.2f} 倍"
        )

    # 6) 雷达图：各群体的“指标百分位均值”（0-100）
    radar_html = "<p class='sub'>暂无数据用于绘制雷达图。</p>"
    if pct_metrics:
        radar_labels = [label for _col, label in pct_metrics]
        fig_r = go.Figure()
        # 固定配色：让“高价值客户群”和“中高价值客户群”区分更明显
        color_map = {
            # 采用高区分度色板（红/蓝/绿/紫/灰），即使填充重叠也容易分辨
            # 这里选更柔和的配色（更浅、更不刺眼），但色相依然区分明显
            "高价值客户群": "#ef5350",          # 柔红
            "中高价值客户群": "#42a5f5",        # 柔蓝
            "中等价值客户群": "#66bb6a",        # 柔绿
            "低价值潜在流失客户群": "#ab47bc",  # 柔紫
            "低价值客户群": "#90a4ae",          # 柔灰蓝
        }
        fallback_colors = [
            "#00bcd4", "#4caf50", "#f44336", "#795548", "#607d8b",
            "#3f51b5", "#8bc34a", "#ff5722",
        ]
        def _hex_to_rgba(hex_color: str, alpha: float) -> str:
            """将 #RRGGBB 转为 rgba(r,g,b,a)，用于给填充设置透明度但不影响边框。"""
            h = hex_color.lstrip("#")
            if len(h) != 6:
                return f"rgba(0,0,0,{alpha})"
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            a = max(0.0, min(1.0, float(alpha)))
            return f"rgba({r},{g},{b},{a})"

        # 最多显示 8 个群体，避免拥挤
        for idx, (seg_name, g) in enumerate(grp):
            if idx >= 8:
                break
            seg_name_str = str(seg_name)
            vals = []
            for col, _label in pct_metrics:
                v = float(pd.to_numeric(g[col], errors="coerce").mean())
                v = 50.0 if pd.isna(v) else max(0.0, min(100.0, v))
                vals.append(v)
            if not vals:
                continue
            color = color_map.get(seg_name_str, fallback_colors[idx % len(fallback_colors)])
            fig_r.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_labels + [radar_labels[0]],
                    name=seg_name_str,
                    mode="lines+markers",
                    fill="toself",
                    # 边框保持不透明并加粗，便于看清每个群体的“面积边界”
                    line=dict(color=color, width=2),
                    # 填充单独设置透明度，避免影响边框清晰度
                    fillcolor=_hex_to_rgba(color, 0.08),
                    # 顶点标记：强调每个维度的落点位置
                    marker=dict(size=6, color=color, line=dict(width=1, color="#ffffff")),
                )
            )
        if fig_r.data:
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            radar_html = fig_r.to_html(include_plotlyjs="cdn", full_html=False)

    # 7) 热力图：各群体关键特征“均值”并标准化到 0-100
    heatmap_html = "<p class='sub'>暂无数据用于绘制热力图。</p>"
    heat_cols = [k for k, _label in metrics if k in df.columns]
    if has_last:
        heat_cols = heat_cols + ["Last_Purchase_Days_Ago"]
    if heat_cols:
        heat_df = df.groupby(seg_col, dropna=False)[heat_cols].mean(numeric_only=True)
        # 最近购买天数越小越好：先反向，让数值越大越“好”，再做 0-100 标准化
        if "Last_Purchase_Days_Ago" in heat_df.columns:
            heat_df["Last_Purchase_Days_Ago"] = heat_df["Last_Purchase_Days_Ago"].max() - heat_df["Last_Purchase_Days_Ago"]
        norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9) * 100
        x_labels = []
        label_map = {k: v for k, v in metrics}
        for c in heat_cols:
            if c == "Last_Purchase_Days_Ago":
                x_labels.append("最近购买(越近越高)")
            else:
                x_labels.append(label_map.get(c, c))
        fig_h = go.Figure(
            data=go.Heatmap(
                z=norm.values,
                x=x_labels,
                y=[str(i) for i in norm.index],
                colorscale="Blues",
                text=heat_df.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        fig_h.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        heatmap_html = fig_h.to_html(include_plotlyjs=False, full_html=False)

    # 8) 论文模块：高价值用户 / 高流失风险用户 / 高互动用户 / 分布统计
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # ---- 高价值用户：按 Total_Sales 前 20% ----
    high_value_def = "高价值用户定义：按总销售额（Total_Sales）从高到低取前 20%（80 分位阈值）。"
    high_value_threshold = None
    high_value_common = {}
    high_value_users = []
    high_value_cluster_dist = []
    if "Total_Sales" in df.columns:
        sales = _num(df["Total_Sales"]).fillna(0.0)
        high_value_threshold = float(sales.quantile(0.80))
        hv = df[sales >= high_value_threshold].copy()
        hv_top = hv.sort_values(by="Total_Sales", ascending=False).head(5)
        for _, r in hv_top.iterrows():
            high_value_users.append(
                {
                    "id": str(r.get("Customer_ID")),
                    "name": str(r.get("Customer_Name") or "-"),
                    "sales": float(_num(pd.Series([r.get("Total_Sales")])).iloc[0] or 0.0),
                    "orders": float(_num(pd.Series([r.get("Total_Orders")])).iloc[0] or 0.0) if "Total_Orders" in hv.columns else 0.0,
                    "aov": float(_num(pd.Series([r.get("Avg_Order_Value")])).iloc[0] or 0.0) if "Avg_Order_Value" in hv.columns else 0.0,
                    "cluster": str(r.get("Cluster_Label") or r.get("Cluster") or "-"),
                    "fav": str(r.get("Favorite_Product") or "-"),
                }
            )

        def _avg(frame: pd.DataFrame, col: str) -> float | None:
            if col not in frame.columns:
                return None
            v = float(_num(frame[col]).mean())
            return None if pd.isna(v) else v

        def _med(frame: pd.DataFrame, col: str) -> float | None:
            if col not in frame.columns:
                return None
            v = float(_num(frame[col]).median())
            return None if pd.isna(v) else v

        top_fav = {}
        if "Favorite_Product" in hv.columns:
            top_fav = hv["Favorite_Product"].astype(str).value_counts().head(3).to_dict()

        high_value_common = {
            "count": int(len(hv)),
            "pct": round(len(hv) / len(df) * 100.0, 1) if len(df) else 0.0,
            "threshold": round(high_value_threshold, 2),
            "avg_lifetime": _avg(hv, "Customer_Lifetime_Days") or _avg(hv, "Customer_Lifetime"),
            "avg_freq": _avg(hv, "Purchase_Frequency"),
            "median_last_days": _med(hv, "Last_Purchase_Days_Ago"),
            "top_fav": top_fav,
        }

        seg_col_hv = "Cluster_Label" if "Cluster_Label" in hv.columns else ("Cluster" if "Cluster" in hv.columns else None)
        if seg_col_hv and not hv.empty:
            vc = hv[seg_col_hv].astype(str).value_counts(dropna=False)
            for k, cnt in vc.items():
                high_value_cluster_dist.append(
                    {"cluster": str(k), "count": int(cnt), "pct": round(int(cnt) / len(hv) * 100.0, 1)}
                )

    # ---- 高流失风险：阈值规则 + >90 天未购拆解 ----
    churn_def = "高流失风险阈值：购买频率 < 0.035 且 距上次购买天数 > 60；并统计 >90 天未购用户。"
    last_days_series = None
    if "Last_Purchase_Days_Ago" in df.columns:
        last_days_series = _num(df["Last_Purchase_Days_Ago"]).fillna(0.0)
    elif "Days_Since_Last_Purchase" in df.columns:
        last_days_series = _num(df["Days_Since_Last_Purchase"]).fillna(0.0)
    else:
        last_days_series = pd.Series([0.0] * len(df))

    freq_series = _num(df["Purchase_Frequency"]).fillna(0.0) if "Purchase_Frequency" in df.columns else pd.Series([0.0] * len(df))
    churn_rule_count = int(((freq_series < 0.035) & (last_days_series > 60)).sum())

    churn90_df = df[last_days_series > 90].copy()
    churn90_count = int(len(churn90_df))
    churn90_cluster_dist = []
    churn90_details = []
    seg_col_churn = "Cluster_Label" if "Cluster_Label" in churn90_df.columns else ("Cluster" if "Cluster" in churn90_df.columns else None)
    if seg_col_churn and churn90_count:
        vc = churn90_df[seg_col_churn].astype(str).value_counts(dropna=False)
        for k, cnt in vc.items():
            churn90_cluster_dist.append(
                {"cluster": str(k), "count": int(cnt), "pct": round(int(cnt) / churn90_count * 100.0, 1)}
            )

    churn90_show = churn90_df.sort_values(by="Total_Sales", ascending=False).head(10) if "Total_Sales" in churn90_df.columns else churn90_df.head(10)
    for _, r in churn90_show.iterrows():
        churn90_details.append(
            {
                "id": str(r.get("Customer_ID")),
                "name": str(r.get("Customer_Name") or "-"),
                "cluster": str(r.get("Cluster_Label") or r.get("Cluster") or "-"),
                "sales": float(_num(pd.Series([r.get("Total_Sales")])).iloc[0] or 0.0) if "Total_Sales" in df.columns else 0.0,
                "freq": float(_num(pd.Series([r.get("Purchase_Frequency")])).iloc[0] or 0.0) if "Purchase_Frequency" in df.columns else 0.0,
                "last_days": float(_num(pd.Series([r.get("Last_Purchase_Days_Ago", r.get("Days_Since_Last_Purchase"))])).iloc[0] or 0.0),
                "fav": str(r.get("Favorite_Product") or "-"),
            }
        )

    # ---- 高互动用户：按 Total_Engagement_Score 前 20% ----
    engage_def = "高互动用户定义：按互动总分（Total_Engagement_Score）取前 20%（80 分位阈值）。"
    engage_threshold = None
    engage_summary = {}
    engage_cluster_dist = []
    engage_high_value_overlap = None
    if "Total_Engagement_Score" in df.columns:
        eng = _num(df["Total_Engagement_Score"]).fillna(0.0)
        engage_threshold = float(eng.quantile(0.80))
        he = df[eng >= engage_threshold].copy()
        engage_summary = {
            "count": int(len(he)),
            "pct": round(len(he) / len(df) * 100.0, 1) if len(df) else 0.0,
            "threshold": round(engage_threshold, 2),
            "avg_aov": round(float(_num(he["Avg_Order_Value"]).mean()), 2) if "Avg_Order_Value" in he.columns else None,
            "avg_sales": round(float(_num(he["Total_Sales"]).mean()), 2) if "Total_Sales" in he.columns else None,
        }
        seg_col_eng = "Cluster_Label" if "Cluster_Label" in he.columns else ("Cluster" if "Cluster" in he.columns else None)
        if seg_col_eng and not he.empty:
            vc = he[seg_col_eng].astype(str).value_counts(dropna=False)
            for k, cnt in vc.items():
                engage_cluster_dist.append(
                    {"cluster": str(k), "count": int(cnt), "pct": round(int(cnt) / len(he) * 100.0, 1)}
                )
        if "Total_Sales" in he.columns and high_value_threshold is not None and not he.empty:
            he_sales = _num(he["Total_Sales"]).fillna(0.0)
            engage_high_value_overlap = round(float((he_sales >= high_value_threshold).mean()) * 100.0, 1)

    # ---- 用户特征统计：分位数 + 区间分布 ----
    stats_summary = {}
    if "Total_Sales" in df.columns:
        s = _num(df["Total_Sales"]).fillna(0.0)
        stats_summary["sales_quantiles"] = {
            "p25": round(float(s.quantile(0.25)), 2),
            "p50": round(float(s.quantile(0.50)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
        }
        stats_summary["sales_min"] = round(float(s.min()), 2)
        stats_summary["sales_max"] = round(float(s.max()), 2)
        stats_summary["sales_mean"] = round(float(s.mean()), 2)
        stats_summary["sales_median"] = round(float(s.median()), 2)
    if "Total_Orders" in df.columns:
        o = _num(df["Total_Orders"]).fillna(0.0)
        bins = [0, 50, 80, 9999]
        labels_bins = ["0-50", "51-80", "81+"]
        cat = pd.cut(o, bins=bins, labels=labels_bins, right=True, include_lowest=True)
        vc = cat.value_counts(normalize=True).sort_index()
        stats_summary["orders_bins"] = {str(k): round(float(v) * 100.0, 1) for k, v in vc.items()}

    return render_template(
        "segment_overview.html",
        total_users=total_users,
        segment_count=int(len(segments)),
        high_value_count=high_value_count,
        churn_count=churn_count,
        segments=segments,
        validation_notes=validation_notes,
        radar_html=radar_html,
        heatmap_html=heatmap_html,
        high_value_def=high_value_def,
        high_value_common=high_value_common,
        high_value_users=high_value_users,
        high_value_cluster_dist=high_value_cluster_dist,
        churn_def=churn_def,
        churn_rule_count=churn_rule_count,
        churn90_count=churn90_count,
        churn90_cluster_dist=churn90_cluster_dist,
        churn90_details=churn90_details,
        engage_def=engage_def,
        engage_summary=engage_summary,
        engage_cluster_dist=engage_cluster_dist,
        engage_high_value_overlap=engage_high_value_overlap,
        stats_summary=stats_summary,
    )


if __name__ == "__main__":
    # debug=True 方便开发时自动重载
    app.run(host="127.0.0.1", port=5000, debug=True)

