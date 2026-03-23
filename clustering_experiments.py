"""
K-Means 与 DBSCAN 聚类算法对比实验脚本

对应开题报告：聚类算法实现与调优、算法对比（Silhouette、CH、运行效率、业务解读性）

用法：
    python clustering_experiments.py

输出：
    - 控制台打印：K-Means 手肘法/轮廓系数、DBSCAN 网格搜索结果、两种算法对比表
    - 可选保存聚类结果到 CSV
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score


DATA_CSV = "customer_features_rfmbc.csv"
# 用 Silhouette（轮廓系数）在 K∈{3,4,5,6} 上选出的最优 K=3。
# 这里用于“算法对比展示”的固定 K；不影响 build_rfmbc_features.py 里最终生成使用的 K。
K_FOR_COMPARE = 3

# 与 notebook 中保持一致的“标准特征名”到多个可能列名的映射，便于兼容不同版本的数据
FEATURE_ALIASES = {
    # 开题落地：RFMB-C 12 个核心特征（Min-Max 后的 _MM 列）
    "R_Recency_Days_MM": ["R_Recency_Days_MM"],
    "R_Avg_Interval_Days_MM": ["R_Avg_Interval_Days_MM"],
    "F_Orders_30d_MM": ["F_Orders_30d_MM"],
    "F_Orders_90d_MM": ["F_Orders_90d_MM"],
    "M_Sales_30d_MM": ["M_Sales_30d_MM"],
    "M_AOV_30d_MM": ["M_AOV_30d_MM"],
    "M_Sales_Var_90d_MM": ["M_Sales_Var_90d_MM"],
    "B_Browse_Count_30d_MM": ["B_Browse_Count_30d_MM"],
    "B_Avg_Browse_Time_30d_MM": ["B_Avg_Browse_Time_30d_MM"],
    "B_Depth_30d_MM": ["B_Depth_30d_MM"],
    "C_Top1_Ratio_MM": ["C_Top1_Ratio_MM"],
    "C_Top3_Coverage_MM": ["C_Top3_Coverage_MM"],

    # 兼容旧数据集（若仍使用旧的 customer_clusters_simple.csv）
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


def get_available_features(df: pd.DataFrame) -> list[str]:
    """解析 DataFrame 列名，返回在当前数据中实际存在的聚类特征列。"""
    cols = list(df.columns)
    available = []
    for canonical, aliases in FEATURE_ALIASES.items():
        for a in aliases:
            if a in cols:
                available.append(a)
                break
    return available


def prepare_X(df: pd.DataFrame, available: list[str]) -> tuple[np.ndarray, MinMaxScaler]:
    """根据可用特征构造聚类输入矩阵：补全缺失值并做 Min-Max 标准化到 [0,1]。"""
    X = df[available].copy()
    X = X.fillna(X.median())
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_kmeans_experiments(X: np.ndarray) -> dict:
    """在多个 K 值上运行 K-Means，记录 SSE / 轮廓系数 / CH 指标，帮助确定最优 K。"""
    k_range = [3, 4, 5, 6]
    sse_list = []
    sil_list = []
    ch_list = []
    times = []

    for k in k_range:
        t0 = time.perf_counter()
        # 固定随机种子，指定 n_init 次数，保证结果可复现且稳定
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        sse = km.inertia_
        sse_list.append(sse)

        n_clusters = len(np.unique(labels[labels >= 0]))
        # 至少需要 2 个簇且样本数大于簇数，轮廓系数和 CH 指标才有意义
        if n_clusters > 1 and n_clusters < len(X):
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil = np.nan
            ch = np.nan
        sil_list.append(sil)
        ch_list.append(ch)

    best_sil_idx = int(np.nanargmax(sil_list))
    best_k = k_range[best_sil_idx]

    return {
        "k_range": k_range,
        "sse": sse_list,
        "silhouette": sil_list,
        "calinski_harabasz": ch_list,
        "time_sec": times,
        "best_k": best_k,
        "best_silhouette": sil_list[best_sil_idx],
    }


def run_dbscan_grid(X: np.ndarray, eps_range: list[float], min_samples_range: list[int]) -> dict:
    """在给定 eps / min_samples 网格上运行 DBSCAN，评估聚类质量与噪声比例。"""
    results = []
    for eps in eps_range:
        for min_samples in min_samples_range:
            t0 = time.perf_counter()
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            elapsed = time.perf_counter() - t0

            n_noise = (labels == -1).sum()
            noise_ratio = n_noise / len(labels) * 100
            n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                mask = labels >= 0
                if mask.sum() > n_clusters:
                    sil = silhouette_score(X[mask], labels[mask])
                    ch = calinski_harabasz_score(X[mask], labels[mask])
                else:
                    sil = np.nan
                    ch = np.nan
            else:
                sil = np.nan
                ch = np.nan

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "noise_ratio_pct": noise_ratio,
                "n_clusters": n_clusters,
                "time_sec": elapsed,
            })

    # 选择：优先多簇(>1)、噪声 < 10%、轮廓系数尽量高的组合
    valid = [
        r for r in results
        if r["n_clusters"] >= 2 and r["noise_ratio_pct"] < 10 and not np.isnan(r["silhouette"])
    ]
    if valid:
        best = max(valid, key=lambda r: r["silhouette"])
    else:
        # 退而求其次：只要多簇
        multi = [r for r in results if r["n_clusters"] >= 2]
        best = max(multi, key=lambda r: r["silhouette"]) if multi else results[0]

    return {"results": results, "best": best}


def compare_algorithms(X: np.ndarray, k_best: int = 5, db_eps: float = 0.5, db_min: int = 5) -> dict:
    """用同一批数据，对比 K-Means 与 DBSCAN 的精度、效率和实际可用性。"""
    # K-Means
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k_best, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_time = time.perf_counter() - t0
    km_sil = silhouette_score(X, km_labels)
    km_ch = calinski_harabasz_score(X, km_labels)

    # DBSCAN
    t0 = time.perf_counter()
    db = DBSCAN(eps=db_eps, min_samples=db_min)
    db_labels = db.fit_predict(X)
    db_time = time.perf_counter() - t0
    mask = db_labels >= 0
    n_clusters = len(np.unique(db_labels[mask]))
    if n_clusters >= 2 and mask.sum() > n_clusters:
        db_sil = silhouette_score(X[mask], db_labels[mask])
        db_ch = calinski_harabasz_score(X[mask], db_labels[mask])
    else:
        db_sil = float("nan")
        db_ch = float("nan")
    db_noise = (db_labels == -1).sum() / len(db_labels) * 100

    return {
        "kmeans": {"silhouette": km_sil, "ch": km_ch, "time_sec": km_time, "n_clusters": k_best},
        "dbscan": {
            "silhouette": db_sil,
            "ch": db_ch,
            "time_sec": db_time,
            "n_clusters": len(np.unique(db_labels[db_labels >= 0])),
            "noise_ratio_pct": db_noise,
        },
    }


def main() -> None:
    print("=" * 60)
    print("K-Means 与 DBSCAN 聚类算法对比实验")
    print("=" * 60)

    # 1) 读入原始 CSV
    df = pd.read_csv(DATA_CSV)
    available = get_available_features(df)
    if not available:
        raise RuntimeError(f"未在 {DATA_CSV} 中找到可用聚类特征列")
    print(f"\n可用特征 ({len(available)} 个): {available}")

    # 2) 预处理特征：缺失值填充 + Min-Max 标准化
    X, _ = prepare_X(df, available)
    n_samples = X.shape[0]
    print(f"样本数: {n_samples}\n")

    # 3.1 K-Means 实验：遍历多个 K 值，记录指标，找出最优 K
    print("-" * 60)
    print("1. K-Means 调优（手肘法 + 轮廓系数）")
    print("-" * 60)
    km_res = run_kmeans_experiments(X)
    for i, k in enumerate(km_res["k_range"]):
        print(f"  K={k}: SSE={km_res['sse'][i]:.0f}, Silhouette={km_res['silhouette'][i]:.4f}, "
              f"CH={km_res['calinski_harabasz'][i]:.2f}, 耗时={km_res['time_sec'][i]:.3f}s")
    k_use = K_FOR_COMPARE if K_FOR_COMPARE is not None else km_res["best_k"]
    print(f"  推荐 K = {km_res['best_k']} (轮廓系数最高)")
    print(f"  比较实验将使用 K = {k_use}。\n")

    # 3.2 DBSCAN 网格搜索：在一小段 eps / min_samples 网格上搜索较优组合
    print("-" * 60)
    print("2. DBSCAN 网格搜索（eps=0.3~0.7, min_samples=3~7）")
    print("-" * 60)
    db_res = run_dbscan_grid(X, eps_range=[0.3, 0.4, 0.5, 0.6, 0.7], min_samples_range=[3, 5, 7])
    for r in db_res["results"][:10]:  # 只打印前 10 组
        print(f"  eps={r['eps']}, min_samples={r['min_samples']}: "
              f"Sil={r['silhouette']:.4f}, 噪声={r['noise_ratio_pct']:.1f}%, 簇数={r['n_clusters']}")
    b = db_res["best"]
    print(f"\n  推荐参数: eps={b['eps']}, min_samples={b['min_samples']} "
          f"(Sil={b['silhouette']:.4f}, 噪声={b['noise_ratio_pct']:.1f}%)\n")

    # 3.3 算法对比：用推荐参数分别跑一次 K-Means 与 DBSCAN，并输出关键指标
    print("-" * 60)
    print("3. 算法对比（精度、效率、业务可解读性）")
    print("-" * 60)
    cmp = compare_algorithms(X, k_best=k_use, db_eps=b["eps"], db_min=b["min_samples"])
    print("  K-Means:")
    print(f"    Silhouette={cmp['kmeans']['silhouette']:.4f}, CH={cmp['kmeans']['ch']:.2f}, "
          f"耗时={cmp['kmeans']['time_sec']:.3f}s, 簇数={cmp['kmeans']['n_clusters']}")
    print("  DBSCAN:")
    db_sil = cmp["dbscan"]["silhouette"]
    db_ch = cmp["dbscan"]["ch"]
    sil_str = f"{db_sil:.4f}" if not np.isnan(db_sil) else "N/A"
    ch_str = f"{db_ch:.2f}" if not np.isnan(db_ch) else "N/A"
    print(f"    Silhouette={sil_str}, CH={ch_str}, "
          f"耗时={cmp['dbscan']['time_sec']:.3f}s, 簇数={cmp['dbscan']['n_clusters']}, "
          f"噪声={cmp['dbscan']['noise_ratio_pct']:.1f}%")
    print("\n  业务解读性: K-Means 簇数固定、标签稳定，适合直接定义 K 类用户群体；"
          "DBSCAN 自动发现簇数，噪声点需单独处理。")
    print("=" * 60)


if __name__ == "__main__":
    main()
