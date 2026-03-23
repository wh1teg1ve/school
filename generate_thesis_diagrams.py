"""
论文用图生成脚本：算法流程图 + RFM-BC 特征体系示意图

依赖：matplotlib（建议 pip install matplotlib）

用法：
    python generate_thesis_diagrams.py

输出目录：figures/（自动创建）
    - thesis_01_overall_pipeline.png   整体技术路线
    - thesis_02_kmeans_flow.png        KMeans 流程
    - thesis_03_dbscan_flow.png        DBSCAN 流程
    - thesis_04_rfmbc_features.png     RFM-BC 特征体系
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DPI = 300


def _setup_chinese_font() -> None:
    """尽量在 Windows 上正确显示中文。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    fs: int = 8,
    face: str = "#e3f2fd",
    edge: str = "#1565c0",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.15,
        zorder=4,
    )


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    rad: float = 0.12,
    zorder: int = 2,
) -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        color="#333333",
        linewidth=1.0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)


def fig_overall_pipeline() -> None:
    """图1：整体技术路线（自上而下）。"""
    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("图1  电商用户分群整体技术路线", fontsize=12, fontweight="bold", pad=12)

    bw, bh = 6.2, 0.55
    cx = 1.9
    ys = [
        (13.1, "数据来源：E_commerce.csv（订单、浏览、互动等）"),
        (12.35, "预处理：时间解析、金额清洗、缺失值填充、Ref_Date 统一口径"),
        (11.6, "RFM-BC 特征：R/F/M/B/C 共 12 维 → customer_features_rfmbc.csv"),
        (10.85, "Min-Max 标准化 [0,1]；R 类天数特征反向（小→优）"),
        (10.1, "聚类：KMeans（K=3）"),
        (9.35, "对比实验：DBSCAN（eps / min_samples 网格）"),
        (8.6, "业务命名：按流失风险 + 销售额映射 Cluster_Label"),
        (7.85, "评估：肘部法、轮廓系数、Calinski-Harabasz"),
        (7.1, "系统：Flask（首页 / 详情 / 群体总览 / algo-compare）"),
    ]
    # 回到“整齐竖直对齐”的版式：不做 zigzag，避免出现你说的歪歪扭扭
    for y, t in ys:
        _box(ax, (cx, y), bw, bh, t, fs=8)

    # 竖向箭头：从上框顶边到下框底边（加更大间隙，避免箭头穿过框）
    for i in range(len(ys) - 1):
        y_from = ys[i][0] + bh + 0.05
        y_to = ys[i + 1][0] - 0.05
        x1 = cx + bw / 2
        _arrow(ax, x1, y_from, x1, y_to, rad=0.08)

    fig.savefig(
        os.path.join(OUT_DIR, "thesis_01_overall_pipeline.png"),
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def fig_kmeans_flow_simple() -> None:
    """图2 简化版：无循环箭头，避免排版问题，更适合 Word。"""
    fig, ax = plt.subplots(figsize=(6.5, 9), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    ax.set_title("图2  KMeans 聚类算法流程", fontsize=12, fontweight="bold", pad=12)

    w, h = 5.6, 0.72
    x = 2.15
    steps = [
        (9.2, "① 开始"),
        (8.25, "② 输入标准化特征矩阵 X\n（12 维 RFM-BC 的 MinMax 结果）"),
        (7.2, "③ 初始化 K 个簇中心（指定 K、随机种子）"),
        (6.15, "④ 将每个样本分配到距最近中心所在簇"),
        (5.1, "⑤ 更新簇中心为簇内样本均值"),
        (4.05, "⑥ 重复④⑤直至收敛\n（中心变化小于阈值或达到最大迭代）"),
        (2.95, "⑦ 输出簇标签 Cluster、簇中心与 SSE"),
        (1.85, "⑧ 可选：计算轮廓系数、CH 指数"),
        (0.75, "⑨ 结束"),
    ]
    # 统一对齐，避免歪斜观感
    for y, t in steps:
        _box(ax, (x, y), w, h, t, fs=8)

    for i in range(len(steps) - 1):
        _arrow(
            ax,
            x + w / 2,
            steps[i][0] + h + 0.05,
            x + w / 2,
            steps[i + 1][0] - 0.05,
            rad=0.08,
        )

    fig.savefig(
        os.path.join(OUT_DIR, "thesis_02_kmeans_flow.png"),
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def fig_dbscan_flow() -> None:
    """图3：DBSCAN 流程。"""
    fig, ax = plt.subplots(figsize=(6.5, 7.5), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("图3  DBSCAN 聚类算法流程（简化）", fontsize=12, fontweight="bold", pad=12)

    w, h = 5.6, 0.72
    x = 2.15
    steps = [
        (7.8, "① 开始"),
        (6.85, "② 输入标准化特征 X\n参数 eps、min_samples"),
        (5.9, "③ 识别核心点\n（eps 邻域内样本数 ≥ min_samples）"),
        (4.95, "④ 从核心点密度可达扩张形成簇"),
        (4.0, "⑤ 边界点归入相邻簇；无法归入者标为噪声 -1"),
        (3.05, "⑥ 输出簇标签与噪声比例"),
        (2.1, "⑦ 多簇时：对非噪声样本可计算轮廓系数、CH"),
        (1.15, "⑧ 结束"),
    ]
    for y, t in steps:
        _box(ax, (x, y), w, h, t, fs=8, face="#f3e5f5", edge="#6a1b9a")

    for i in range(len(steps) - 1):
        _arrow(
            ax,
            x + w / 2,
            steps[i][0] + h + 0.05,
            x + w / 2,
            steps[i + 1][0] - 0.05,
            rad=0.08,
        )

    fig.savefig(
        os.path.join(OUT_DIR, "thesis_03_dbscan_flow.png"),
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def fig_rfmbc_tree() -> None:
    """图4：RFM-BC 特征体系（与工程列名一致）。"""
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=100)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("图4  RFM-BC 多维特征体系示意图", fontsize=12, fontweight="bold", pad=12)

    # 根
    _box(ax, (5.0, 6.85), 4.0, 0.55, "RFM-BC 用户特征体系", fs=9, face="#fff8e1", edge="#f57f17")

    # 五个一级
    branches = [
        (0.2, 5.5, 2.5, 1.15, "R 最近性\nR_Recency_Days\nR_Avg_Interval_Days", "#ffebee"),
        (2.95, 5.5, 2.5, 1.15, "F 频次\nF_Orders_30d\nF_Orders_90d", "#e8f5e9"),
        (5.7, 5.5, 2.5, 1.15, "M 金额\nM_Sales_30d\nM_AOV_30d\nM_Sales_Var_90d", "#e3f2fd"),
        (8.45, 5.5, 2.5, 1.15, "B 浏览\nB_Browse_Count_30d\nB_Avg_Browse_Time_30d\nB_Depth_30d", "#f3e5f5"),
        (11.2, 5.5, 2.5, 1.15, "C 品类\nC_Top1_Ratio\nC_Top3_Coverage", "#eceff1"),
    ]
    for bx, by, bw, bh, txt, fc in branches:
        _box(ax, (bx, by), bw, bh, txt, fs=7, face=fc, edge="#37474f")

    root_cx, root_y = 7.0, 6.85
    for bx, by, bw, bh, _, _ in branches:
        # 图4箭头需要“盖在框之上”，否则容易被框边缘遮住导致看不到
        _arrow(
            ax,
            root_cx,
            root_y - 0.02,
            bx + bw / 2,
            by + bh + 0.03,
            rad=0.06,
            zorder=10,
        )

    note = (
        "聚类输入：上述 12 列经 MinMaxScaler 得到 *_MM 列；"
        "R_Recency_Days_MM 与 R_Avg_Interval_Days_MM 再作 1−x 方向调整（天数越小越优）。"
    )
    _box(ax, (0.3, 0.35), 13.4, 0.85, note, fs=7, face="#fafafa", edge="#9e9e9e")

    fig.savefig(
        os.path.join(OUT_DIR, "thesis_04_rfmbc_features.png"),
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    _setup_chinese_font()

    fig_overall_pipeline()
    fig_kmeans_flow_simple()  # 用简化版，避免循环箭头在 Word 里难对齐
    fig_dbscan_flow()
    fig_rfmbc_tree()

    print(f"[完成] 已生成 4 张图到目录: {OUT_DIR}")
    for name in (
        "thesis_01_overall_pipeline.png",
        "thesis_02_kmeans_flow.png",
        "thesis_03_dbscan_flow.png",
        "thesis_04_rfmbc_features.png",
    ):
        p = os.path.join(OUT_DIR, name)
        print(f"  - {p}")


if __name__ == "__main__":
    main()
