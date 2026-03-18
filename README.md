# 电商用户画像系统（Flask + Pandas + Plotly）

本项目是一个面向课程/实验的电商用户画像与分群展示系统，支持从 **CSV** 或 **MySQL** 读取客户数据，在网页端提供客户列表、客户详情画像、聚类分布可视化与多客户对比等功能。

## 功能概览

- **首页仪表盘**：
  - 客户列表（支持按客户 ID / 群体筛选）
  - 点击表头排序（如：活跃度、订单数、客单价等）
  - 聚类分布饼图
  - 跨分页/跨筛选的多选客户对比（本地 `localStorage` 记忆）
  - 随机客户跳转
  - 筛选后显示：该群体核心特征概览卡片（业务命名、流失风险、与整体差异）
- **客户详情页**：
  - 指标百分位条（红线标记位置）
  - 雷达图、散点图、订单趋势图
  - 客户分析要点 + 推荐营销方案
  - 综合评分 + 运营优先级
- **多客户对比页**：
  - 多客户关键指标对比表
  - 多客户雷达图
  - “保留第一位 + 随机选择后两位”随机对比
- **群体总览页（RFM-BC）**：`/segment-overview`
  - 群体对比图（雷达图 + 热力图）
  - 群体概览卡片（无需横向滚动）
  - 聚类差异合理性验证（倍数差异）
  - 高价值用户 / 高流失风险用户 / 高互动用户 / 分布统计（论文模块）
- **算法对比页（论文用）**：`/algo-compare`
  - K-Means：SSE 手肘图、Silhouette 曲线、CH 指标与耗时
  - DBSCAN：eps/min_samples 网格搜索 + 噪声比例
  - 自动生成“论文可直接引用的总结”
- **数据源切换**：同一套页面可选择 **CSV** 或 **MySQL** 数据源。

## 项目结构（主要文件）

- `flask_app.py`：Flask 主程序（路由、数据处理、模板、图表生成）
- `templates/`：Flask 标准模板目录（HTML 模板）
- `static/`：静态资源目录（CSS / JS）
- `start_all.py`：一键启动脚本（安装依赖 → 可选导入 MySQL → 启动 Flask → 打开浏览器）
- `build_rfmbc_features.py`：从 `E_commerce.csv` 生成 RFMB-C 12 核心特征与画像字段，输出 `customer_features_rfmbc.csv`
- `import_csv_to_mysql.py`：将 `customer_features_rfmbc.csv` 导入 MySQL
- `clustering_experiments.py`：K-Means 与 DBSCAN 聚类算法对比实验
- `run_visualizations.py`：独立生成 RFM-BC 雷达图与聚类特征热力图 HTML
- `customer_features_rfmbc.csv`：网页展示默认使用的数据（若使用 CSV 模式）

## 主要页面入口

- 首页：`/`
- 群体总览（RFM-BC）：`/segment-overview`
- 算法对比（K-Means vs DBSCAN）：`/algo-compare`
- 多客户对比：`/compare`
- 随机客户：`/user/random`

## 环境要求

- Python **3.8+**
- 建议使用虚拟环境（可选）

安装依赖：

```bash
pip install -r requirements.txt
```

## 启动方式

### 方式 1：直接启动（推荐入门）

确保项目根目录存在 `customer_features_rfmbc.csv`，然后运行：

```bash
python flask_app.py
```

浏览器打开：

- `http://127.0.0.1:5000`

说明：

- 现在页面使用 **Flask 标准模板**，请确保项目中包含 `templates/` 与 `static/` 目录，否则启动后会出现“找不到模板/静态资源”的错误。

### 方式 2：一键启动（推荐演示/交付）

```bash
python start_all.py
```

该脚本会：

- 安装依赖（`requirements.txt`）
- 若检测到 `customer_features_rfmbc.csv`：尝试导入 MySQL（调用 `import_csv_to_mysql.py`）
- 启动 `flask_app.py`
- 自动打开浏览器访问首页

### 方式 3：使用 MySQL 作为数据源

1. 确保 MySQL 已安装并运行
2. 导入 CSV 到 MySQL：

```bash
python import_csv_to_mysql.py
```

默认会写入表 `customer_profiles_rfmbc`（可用环境变量 `MYSQL_TABLE` 自定义表名）。

3. 启用 MySQL 数据源后启动 Flask（Windows PowerShell 示例）：

```powershell
$env:USE_MYSQL="1"
python flask_app.py
```

Linux/Mac 示例：

```bash
export USE_MYSQL=1
python flask_app.py
```

## 配置项（环境变量）

- `USE_MYSQL`：是否启用 MySQL（`1/true/yes` 表示启用）
- `MYSQL_HOST`：默认 `127.0.0.1`
- `MYSQL_PORT`：默认 `3306`
- `MYSQL_USER`：默认 `root`
- `MYSQL_PASSWORD`：默认空
- `MYSQL_DATABASE`：默认 `customer_profile`
- `MYSQL_TABLE`：默认 `customer_profiles_rfmbc`
- `RESET_TABLE`：导入脚本是否先清空表（默认 `1`，表示 `TRUNCATE` 后再导入）

## 常用脚本

- 生成系统默认数据源（RFMB-C 12 特征 + 画像字段 + 聚类标签）：

```bash
python build_rfmbc_features.py
```

产物（部分）：
- `customer_features_rfmbc.csv`：用户级特征表（含 Ref_Date / Last_Purchase_Date 等可复现口径字段）
- `data_preprocess_report.json`：数据预处理摘要（缺失值统计、参考日期等，便于论文引用）

- 生成聚类对比实验结果：

```bash
python clustering_experiments.py
```

- 生成可视化 HTML（输出到项目根目录）：

```bash
python run_visualizations.py
```

输出：

- `rfm_bc_radar.html`
- `cluster_feature_heatmap.html`

## 常见问题

- **Q：启动报错找不到 `customer_features_rfmbc.csv`？**  
  - **A**：先运行 `python build_rfmbc_features.py` 从 `E_commerce.csv` 生成该文件，然后再启动 Flask。

- **Q：端口 5000 被占用？**  
  - **A**：修改 `flask_app.py` 底部 `app.run(..., port=5000)` 的端口，例如 `5001`。

- **Q：启动后提示找不到模板（TemplateNotFound）？**  
  - **A**：确认项目根目录包含 `templates/` 目录（例如 `templates/index.html`），并且运行时工作目录在项目根目录。

- **Q：MySQL 连接失败？**  
  - **A**：确认 MySQL 服务已启动、账号密码正确、端口可访问；必要时设置 `MYSQL_HOST/USER/PASSWORD` 环境变量。

