"""
从 E_commerce.csv 生成 RFMB-C 12 核心特征（Min-Max 到 [0,1]）并输出 customer_features_rfmbc.csv。

说明：
- 该脚本与 clean.ipynb 中“RFMB-C 12 核心特征”计算保持一致，便于不打开 notebook 也能重算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_sales(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    import json

    # low_memory=False 避免混合类型列导致 dtype 推断警告
    raw = pd.read_csv("E_commerce.csv", low_memory=False)

    raw["Order Date"] = pd.to_datetime(raw["Order Date"], errors="coerce")
    raw["Sales"] = _clean_sales(raw["Sales"])
    raw["Browsing Time (min)"] = pd.to_numeric(raw["Browsing Time (min)"], errors="coerce")

    # ---------- 数据预处理摘要（用于论文“数据来源与预处理”可复现说明） ----------
    preprocess_report = {
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "missing_by_col_before": {k: int(v) for k, v in raw.isna().sum().to_dict().items()},
    }

    raw["Sales"] = raw["Sales"].fillna(raw["Sales"].median())
    raw["Browsing Time (min)"] = raw["Browsing Time (min)"].fillna(raw["Browsing Time (min)"].median())
    raw["Order Date"] = raw["Order Date"].fillna(raw["Order Date"].mode(dropna=True)[0])

    preprocess_report["missing_total_after_core_fill"] = int(raw[["Sales", "Browsing Time (min)", "Order Date"]].isna().sum().sum())

    id_col = "Customer ID"
    name_col = "Customer Name"
    date_col = "Order Date"
    cat_col = "Product Category"
    prod_col = "Product"

    ref_date = raw[date_col].max()
    g = raw.sort_values([id_col, date_col]).copy()

    # 订单月份偏好（行为模式）：取全周期订单月份众数
    g["_order_month"] = g[date_col].dt.month.astype("Int64")
    pref_order_month = (
        g.groupby(id_col)["_order_month"]
        .agg(lambda x: int(x.mode(dropna=True).iloc[0]) if not x.mode(dropna=True).empty else None)
        .astype("Int64")
    )

    # ---------- 画像基础字段（供 Flask 展示使用） ----------
    first_date = g.groupby(id_col)[date_col].min()
    last_date = g.groupby(id_col)[date_col].max()
    customer_lifetime_days = (last_date - first_date).dt.days.clip(lower=0).astype(float)
    days_since_last_purchase = (ref_date - last_date).dt.days.astype(float)

    total_orders = g.groupby(id_col).size().astype(float)
    total_sales = g.groupby(id_col)["Sales"].sum().astype(float)
    avg_order_value = (total_sales / total_orders.replace(0, np.nan)).fillna(0.0).astype(float)

    # 行为计数（若列不存在则按 0 处理）
    for col in ["Like", "Share", "Add to Cart"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).astype(float)
        else:
            g[col] = 0.0
    total_likes = g.groupby(id_col)["Like"].sum().astype(float)
    total_shares = g.groupby(id_col)["Share"].sum().astype(float)
    total_add_to_cart = g.groupby(id_col)["Add to Cart"].sum().astype(float)
    total_engagement_score = (total_likes + total_shares + total_add_to_cart).astype(float)

    avg_browsing_time = g.groupby(id_col)["Browsing Time (min)"].mean().astype(float)
    unique_products_purchased = g.groupby(id_col)[prod_col].nunique().astype(float)

    # 偏好品类：全周期 TOP1 品类（用于详情页展示）
    fav_cat = (
        g.groupby([id_col, cat_col]).size().rename("cnt").reset_index()
        .sort_values([id_col, "cnt"], ascending=[True, False])
        .groupby(id_col).head(1).set_index(id_col)[cat_col]
        .astype(str)
    )

    # 偏好品类 Top3（行为模式对比）：用“品类:占比%”拼成字符串，便于页面展示
    cat_counts2 = g.groupby([id_col, cat_col]).size().rename("cnt").reset_index()
    cat_totals2 = cat_counts2.groupby(id_col)["cnt"].sum()
    cat_sorted2 = cat_counts2.sort_values([id_col, "cnt"], ascending=[True, False])
    top3 = cat_sorted2.groupby(id_col).head(3).copy()
    top3["ratio"] = top3.apply(lambda r: float(r["cnt"]) / float(cat_totals2.loc[r[id_col]]) if r[id_col] in cat_totals2.index else 0.0, axis=1)
    top3["piece"] = top3[cat_col].astype(str) + ":" + (top3["ratio"] * 100).round(1).astype(str) + "%"
    top3_str = top3.groupby(id_col)["piece"].apply(lambda s: "、".join(s.tolist()))

    purchase_frequency = (total_orders / (customer_lifetime_days.replace(0, np.nan))).fillna(0.0).astype(float)

    r_recency_days = days_since_last_purchase

    intervals = g.groupby(id_col)[date_col].diff().dt.days
    r_avg_interval_days = intervals.groupby(g[id_col]).mean().astype(float)

    mask_30 = g[date_col] >= (ref_date - pd.Timedelta(days=30))
    mask_90 = g[date_col] >= (ref_date - pd.Timedelta(days=90))

    f_orders_30d = g[mask_30].groupby(id_col).size().astype(float)
    f_orders_90d = g[mask_90].groupby(id_col).size().astype(float)

    m_sales_30d = g[mask_30].groupby(id_col)["Sales"].sum().astype(float)

    aov_all = g.groupby(id_col)["Sales"].mean().astype(float)
    aov_30d = g[mask_30].groupby(id_col)["Sales"].mean().astype(float)
    m_aov_30d = aov_30d.reindex(aov_all.index).fillna(aov_all)

    m_sales_var_90d = g[mask_90].groupby(id_col)["Sales"].var(ddof=0).astype(float)

    b_browse_count_30d = g[mask_30].groupby(id_col).size().astype(float)
    b_avg_browse_time_30d = g[mask_30].groupby(id_col)["Browsing Time (min)"].mean().astype(float)
    b_depth_30d = g[mask_30].groupby(id_col)[prod_col].nunique().astype(float)

    cat_counts = g.groupby([id_col, cat_col]).size().rename("cnt").reset_index()
    cat_totals = cat_counts.groupby(id_col)["cnt"].sum()
    cat_sorted = cat_counts.sort_values([id_col, "cnt"], ascending=[True, False])
    cat_top1 = cat_sorted.groupby(id_col).head(1).set_index(id_col)["cnt"]
    c_top1_ratio = (cat_top1 / cat_totals).astype(float)
    top3_sum = cat_sorted.groupby(id_col).head(3).groupby(id_col)["cnt"].sum()
    c_top3_coverage = (top3_sum / cat_totals).astype(float)

    features = (
        pd.DataFrame(
            {
                # 基础画像字段（用于网页展示）
                "Total_Orders": total_orders,
                "Total_Sales": total_sales,
                "Avg_Order_Value": avg_order_value,
                "Customer_Lifetime_Days": customer_lifetime_days,
                "Days_Since_Last_Purchase": days_since_last_purchase,
                # 论文/复现口径：用同一个 Ref_Date 计算时间窗口，避免“今天”导致结果漂移
                # Ref_Date：全量数据中的最大订单日期（全局参考日）
                "Ref_Date": pd.Series([ref_date] * len(total_orders), index=total_orders.index),
                # Last_Purchase_Date：每个客户最后一次下单日期（真实日期）
                "Last_Purchase_Date": last_date,
                # 行为模式字段（用于群体对比/论文阐述）
                "Pref_Order_Month": pref_order_month,
                "Purchase_Frequency": purchase_frequency,
                "Avg_Browsing_Time": avg_browsing_time,
                "Unique_Products_Purchased": unique_products_purchased,
                "Favorite_Product": fav_cat,
                "Favorite_Top3": top3_str,
                "Total_Likes": total_likes,
                "Total_Shares": total_shares,
                "Total_Add_to_Cart": total_add_to_cart,
                "Total_Engagement_Score": total_engagement_score,

                "R_Recency_Days": r_recency_days,
                "R_Avg_Interval_Days": r_avg_interval_days,
                "F_Orders_30d": f_orders_30d,
                "F_Orders_90d": f_orders_90d,
                "M_Sales_30d": m_sales_30d,
                "M_AOV_30d": m_aov_30d,
                "M_Sales_Var_90d": m_sales_var_90d,
                "B_Browse_Count_30d": b_browse_count_30d,
                "B_Avg_Browse_Time_30d": b_avg_browse_time_30d,
                "B_Depth_30d": b_depth_30d,
                "C_Top1_Ratio": c_top1_ratio,
                "C_Top3_Coverage": c_top3_coverage,
            }
        )
        .reset_index()
        .rename(columns={id_col: "Customer_ID"})
    )

    name_map = g.groupby(id_col)[name_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    features["Customer_Name"] = features["Customer_ID"].map(name_map).fillna("Unknown")

    fill_zero_cols = [
        "R_Avg_Interval_Days",
        "F_Orders_30d",
        "F_Orders_90d",
        "M_Sales_30d",
        "M_Sales_Var_90d",
        "B_Browse_Count_30d",
        "B_Avg_Browse_Time_30d",
        "B_Depth_30d",
        "C_Top1_Ratio",
        "C_Top3_Coverage",
    ]
    for c in fill_zero_cols:
        features[c] = features[c].fillna(0.0)

    rfmbc_cols = [
        "R_Recency_Days",
        "R_Avg_Interval_Days",
        "F_Orders_30d",
        "F_Orders_90d",
        "M_Sales_30d",
        "M_AOV_30d",
        "M_Sales_Var_90d",
        "B_Browse_Count_30d",
        "B_Avg_Browse_Time_30d",
        "B_Depth_30d",
        "C_Top1_Ratio",
        "C_Top3_Coverage",
    ]

    X = features[rfmbc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_mm = pd.DataFrame(scaler.fit_transform(X), columns=[f"{c}_MM" for c in rfmbc_cols])

    # R 类：天数越小越好 -> 标准化后反向
    X_mm["R_Recency_Days_MM"] = 1.0 - X_mm["R_Recency_Days_MM"]
    X_mm["R_Avg_Interval_Days_MM"] = 1.0 - X_mm["R_Avg_Interval_Days_MM"]

    keep_base = [
        "Customer_ID",
        "Customer_Name",
        "Total_Orders",
        "Total_Sales",
        "Avg_Order_Value",
        "Customer_Lifetime_Days",
        "Days_Since_Last_Purchase",
        "Ref_Date",
        "Last_Purchase_Date",
        "Pref_Order_Month",
        "Purchase_Frequency",
        "Avg_Browsing_Time",
        "Unique_Products_Purchased",
        "Favorite_Product",
        "Favorite_Top3",
        "Total_Likes",
        "Total_Shares",
        "Total_Add_to_Cart",
        "Total_Engagement_Score",
    ]
    out = pd.concat([features[keep_base].reset_index(drop=True), X_mm], axis=1)

    # 使用实验中 Silhouette 最优的 K=3 进行最终分群
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    out["Cluster"] = km.fit_predict(out[[c for c in out.columns if c.endswith("_MM")]].values)

    # 为避免“KMeans 簇编号是任意的导致标签错配”，这里改为数据驱动命名：
    # 1) 先用 Days_Since_Last_Purchase 找出“潜在流失”簇（>60 天未购比例最高）
    # 2) 再在剩余簇里按 Total_Sales（中位数）从高到低分配高/中高/中等/低价值标签
    cluster_stats = (
        out.groupby("Cluster")
        .agg(
            total_sales_median=("Total_Sales", "median"),
            days_since_mean=("Days_Since_Last_Purchase", "mean"),
            churn_rate=("Days_Since_Last_Purchase", lambda s: float((s > 60).mean())),
        )
        .reset_index()
    )

    # 潜在流失簇：优先 churn_rate 最大；如果所有簇 churn_rate 都为 0，则退而选 days_since_mean 最大
    churn_row = cluster_stats.sort_values(
        ["churn_rate", "days_since_mean"], ascending=[False, False]
    ).iloc[0]
    churn_cluster = int(churn_row["Cluster"])
    if float(churn_row["churn_rate"]) <= 0:
        churn_cluster = int(
            cluster_stats.sort_values("days_since_mean", ascending=False).iloc[0]["Cluster"]
        )

    remaining_clusters = [int(c) for c in sorted(out["Cluster"].unique()) if int(c) != churn_cluster]
    remaining_sorted = sorted(
        remaining_clusters,
        key=lambda c: float(
            cluster_stats.loc[cluster_stats["Cluster"] == c, "total_sales_median"].iloc[0]
        ),
        reverse=True,
    )

    mapped_labels: dict[int, str] = {churn_cluster: "低价值潜在流失客户群"}
    other_labels = [
        "高价值客户群",
        "中高价值客户群",
        "中等价值客户群",
        "低价值客户群",
    ]
    for i, c in enumerate(remaining_sorted):
        mapped_labels[int(c)] = other_labels[i] if i < len(other_labels) else str(c)

    out["Cluster_Label"] = out["Cluster"].map(mapped_labels).fillna(out["Cluster"].astype(str))

    out_path = "customer_features_rfmbc.csv"
    out.to_csv(out_path, index=False)
    print(f"已生成：{out_path} shape={out.shape}")

    # 输出预处理报告，便于论文复现引用
    preprocess_report.update(
        {
            "ref_date": str(ref_date.date()) if hasattr(ref_date, "date") else str(ref_date),
            "customers": int(out.shape[0]),
            "output_columns": int(out.shape[1]),
        }
    )
    with open("data_preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_report, f, ensure_ascii=False, indent=2)
    print("已生成：data_preprocess_report.json")


if __name__ == "__main__":
    main()

