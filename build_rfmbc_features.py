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

    # low_memory=False 避免混合类型列导致 dtype 推断警告
    raw = pd.read_csv("E_commerce.csv", low_memory=False)

    raw["Order Date"] = pd.to_datetime(raw["Order Date"], errors="coerce")
    raw["Sales"] = _clean_sales(raw["Sales"])
    raw["Browsing Time (min)"] = pd.to_numeric(raw["Browsing Time (min)"], errors="coerce")

    raw["Sales"] = raw["Sales"].fillna(raw["Sales"].median())
    raw["Browsing Time (min)"] = raw["Browsing Time (min)"].fillna(raw["Browsing Time (min)"].median())
    raw["Order Date"] = raw["Order Date"].fillna(raw["Order Date"].mode(dropna=True)[0])

    id_col = "Customer ID"
    name_col = "Customer Name"
    date_col = "Order Date"
    cat_col = "Product Category"
    prod_col = "Product"

    ref_date = raw[date_col].max()
    g = raw.sort_values([id_col, date_col]).copy()

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
                "Purchase_Frequency": purchase_frequency,
                "Avg_Browsing_Time": avg_browsing_time,
                "Unique_Products_Purchased": unique_products_purchased,
                "Favorite_Product": fav_cat,
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
        "Purchase_Frequency",
        "Avg_Browsing_Time",
        "Unique_Products_Purchased",
        "Favorite_Product",
        "Total_Likes",
        "Total_Shares",
        "Total_Add_to_Cart",
        "Total_Engagement_Score",
    ]
    out = pd.concat([features[keep_base].reset_index(drop=True), X_mm], axis=1)

    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    out["Cluster"] = km.fit_predict(out[[c for c in out.columns if c.endswith("_MM")]].values)
    label_map = {
        0: "高价值客户群",
        1: "低价值客户群",
        2: "中高价值客户群",
        3: "中等价值客户群",
        4: "低价值潜在流失客户群",
    }
    out["Cluster_Label"] = out["Cluster"].map(label_map).fillna(out["Cluster"].astype(str))

    out_path = "customer_features_rfmbc.csv"
    out.to_csv(out_path, index=False)
    print(f"已生成：{out_path} shape={out.shape}")


if __name__ == "__main__":
    main()

