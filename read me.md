# Beijing Multi-Site Air Quality Visualization

本项目使用 **Python** 对北京多站点空气质量数据集（Beijing Multi-Site Air-Quality Data Set）进行数据预处理与可视化分析，包含：

- 数据基本概况查看（`head()`, `info()`, `describe()`）
- 缺失值可视化（`missingno.matrix`）
- 污染物分布直方图
- 时间序列折线图（按站点的 PM2.5 日均变化）
- 按月份的箱线图（boxplot）
- 主要污染物与气象变量的相关性热力图

---

## 1. 数据集说明

- 数据来源：Kaggle 公共数据集  
  **Beijing Multi-Site Air-Quality Data Set**
- 数据内容：北京多个监测站点（station）的空气质量与气象数据  
  - 污染物：`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3` 等  
  - 气象变量：`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM` 等  
  - 时间字段：`year`, `month`, `day`, `hour` 等  
- 本项目中的脚本会：
  - 从压缩包中读取所有站点的 CSV 文件
  - 合并成一个完整数据集
  - 构造时间特征（`datetime`, `date`, `weekday`, `hour_of_day`）

> ⚠️ 请确保数据压缩包文件名为：  
> **`Beijing Multi-Site Air-Quality Data Set.zip`**  
> 并与 Python 脚本放在同一目录下。

---

## 2. 项目文件说明

仓库中主要文件及作用：

- `beijing_air_quality_analysis.py`  
  主脚本：  
  - 读取压缩包中的所有 CSV  
  - 进行基础数据预处理  
  - 生成多种可视化图表到 `figures/` 文件夹

- `Beijing Multi-Site Air-Quality Data Set.zip`（可选是否上传到 GitHub）  
  - 原始空气质量数据压缩包  
  - 脚本运行时会自动从该压缩包中读取数据

- `figures/`（脚本运行后自动生成）  
  - `missing_matrix.png`：缺失值可视化矩阵  
  - `PM2.5_hist.png`：PM2.5 直方图  
  - `PM25_timeseries_Aotizhongxin.png`：奥体中心站 PM2.5 日均时间序列  
  - `PM25_boxplot_month_Aotizhongxin.png`：奥体中心站 PM2.5 按月份箱线图  
  - `correlation_heatmap.png`：主要变量相关性热力图  

---

## 3. 环境依赖

- Python 3.x
- 主要第三方库：
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `missingno`

## 示例图像

### 缺失值可视化

![Missing matrix](figures/missing_matrix.png)

### PM2.5 分布直方图

![PM2.5 histogram](figures/PM2.5_hist.png)

### 相关性热力图

![Correlation heatmap](figures/correlation_heatmap.png)

可以使用以下命令安装依赖：

```bash
pip install pandas numpy matplotlib seaborn missingno
