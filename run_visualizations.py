"""独立运行 RFM-BC 雷达图与热力图，无需打开 notebook。用法：python run_visualizations.py"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

CSV_PATH = "customer_clusters_simple.csv"


def main() -> None:
    """从 CSV 生成 RFM-BC 雷达图与热力图 HTML 文件。"""
    df = pd.read_csv(CSV_PATH)

    # 补全 Cluster_Label（若 CSV 无此列）
    if "Cluster_Label" not in df.columns and "Cluster" in df.columns:
        labels = {
            0: "高价值客户群",
            1: "低价值客户群",
            2: "中高价值客户群",
            3: "中等价值客户群",
            4: "低价值潜在流失客户群",
        }
        df["Cluster_Label"] = df["Cluster"].map(labels)

    cluster_col = "Cluster_Label" if "Cluster_Label" in df.columns else "Cluster"

    # RFM-BC 雷达图
    rfm_cols = {
        "R-最近购买": "Last_Purchase_Days_Ago",
        "F-订单数": "Total_Orders",
        "M-销售额": "Total_Sales",
        "B-互动分": "Total_Engagement_Score",  # 兼容 Engagement_Score
        "C-品类广度": "Unique_Products_Purchased",  # 兼容 Unique_Products_Bought
    }
    rfm_cols = {k: v for k, v in rfm_cols.items() if v in df.columns}
    if "B-互动分" not in rfm_cols and "Engagement_Score" in df.columns:
        rfm_cols["B-互动分"] = "Engagement_Score"
    if not rfm_cols:
        rfm_cols = {
            "销售额": "Total_Sales",
            "订单数": "Total_Orders",
            "客单价": "Avg_Order_Value",
            "购买频率": "Purchase_Frequency",
            "互动分": "Engagement_Score",
        }
        rfm_cols = {k: v for k, v in rfm_cols.items() if v in df.columns}

    grp = df.groupby(cluster_col)[list(rfm_cols.values())].mean()
    if "Last_Purchase_Days_Ago" in rfm_cols.values():
        grp["Last_Purchase_Days_Ago"] = grp["Last_Purchase_Days_Ago"].max() - grp["Last_Purchase_Days_Ago"]
    for c in grp.columns:
        mi, mx = grp[c].min(), grp[c].max()
        grp[c] = (grp[c] - mi) / (mx - mi + 1e-9) * 100

    fig = go.Figure()
    categories = list(rfm_cols.keys())
    for idx, row in grp.iterrows():
        vals = [float(row[c]) for c in rfm_cols.values()]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=str(idx),
                fill="toself",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="各用户群体 RFM-BC 雷达图",
    )
    fig.write_html("rfm_bc_radar.html", include_plotlyjs="cdn")
    print("已生成 rfm_bc_radar.html")

    # 热力图
    feat_cols = [
        c
        for c in [
            "Total_Sales",
            "Total_Orders",
            "Avg_Order_Value",
            "Customer_Lifetime_Days",
            "Purchase_Frequency",
            "Profit_Margin",
            "Total_Engagement_Score",
        ]
        if c in df.columns
    ]
    if not feat_cols:
        feat_cols = [
            c
            for c in [
                "Total_Sales",
                "Total_Orders",
                "Avg_Order_Value",
                "Customer_Lifetime",
                "Purchase_Frequency",
                "Profit_Margin",
                "Engagement_Score",
            ]
            if c in df.columns
        ]
    heat_df = df.groupby(cluster_col)[feat_cols].mean()
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9) * 100
    fig2 = go.Figure(
        data=go.Heatmap(
            z=heat_norm.values,
            x=[c.replace("_", " ") for c in feat_cols],
            y=[str(i) for i in heat_norm.index],
            colorscale="RdYlGn",
            text=heat_df.round(1).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig2.update_layout(
        title="各用户群体关键特征热力图",
        xaxis_title="特征",
        yaxis_title="用户群体",
    )
    fig2.write_html("cluster_feature_heatmap.html", include_plotlyjs="cdn")
    print("已生成 cluster_feature_heatmap.html")


if __name__ == "__main__":
    main()
