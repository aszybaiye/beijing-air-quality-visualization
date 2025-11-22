"""
北京多站点空气质量数据集（Beijing Multi-Site Air-Quality Data Set）数据处理与可视化脚本

说明：
- 使用的数据来自压缩文件：Beijing Multi-Site Air-Quality Data Set.zip
- 自动读取压缩包中的所有站点 CSV 文件并合并
- 进行基础数据预处理和多种可视化分析
- 使用到的可视化方法与课堂 notebook 中类似，包括：
  * 缺失值矩阵（missingno.matrix）
  * 直方图（hist）
  * 折线图（line plot）
  * 箱线图（boxplot）
  * 相关性热力图（seaborn.heatmap）

运行前请确认：
1. 已安装所需库：
   pip install pandas numpy matplotlib seaborn missingno
2. 本脚本与 “Beijing Multi-Site Air-Quality Data Set.zip” 放在同一目录下，
   或者在 main() 中修改 zip_path 为你的实际路径。
"""

import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 一些绘图的全局设置（中文环境可根据需要调整字体）
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def load_beijing_data(zip_path: str) -> pd.DataFrame:
    """
    从压缩文件中读取所有站点的 CSV，并合并成一个 DataFrame。

    参数
    ----
    zip_path : str
        Beijing Multi-Site Air-Quality Data Set.zip 的路径

    返回
    ----
    data : pandas.DataFrame
        合并后的数据集，包含所有站点
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到压缩文件：{zip_path}")

    print(f"正在从压缩文件读取数据：{zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]

        if not csv_files:
            raise ValueError("压缩文件中没有找到任何 CSV 文件。")

        dfs = []
        for fname in csv_files:
            print(f"  读取文件：{fname}")
            with zf.open(fname) as f:
                df = pd.read_csv(f)
                dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"合并后数据形状：{data.shape}")
    return data


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 year, month, day, hour 构造时间相关特征（datetime, date, weekday 等），
    类似于课堂中对时间字段的处理（提取 day、weekday、hour 等）。

    返回处理后的新 DataFrame（不修改原 df）。
    """
    df = df.copy()

    # 构造 datetime 列
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
        df["date"] = df["datetime"].dt.date
        df["weekday"] = df["datetime"].dt.weekday
        df["hour_of_day"] = df["datetime"].dt.hour
    else:
        print("警告：缺少 year/month/day/hour 列，无法构造时间特征。")

    return df


def basic_overview(df: pd.DataFrame) -> None:
    """
    打印基本信息：前几行、info、描述性统计。
    这些操作在课堂 notebook 中经常出现，用于快速了解数据结构。
    """
    print("\n===== 数据前 5 行（head） =====")
    print(df.head())

    print("\n===== df.info() =====")
    print(df.info())

    print("\n===== df.describe()（数值列） =====")
    print(df.describe())


def plot_missing_matrix(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """
    使用 missingno.matrix 进行缺失值可视化（抽样），
    对应 notebook 中的缺失值可视化方法。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 为了节省时间和内存，只抽样一部分行进行展示
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    msno.matrix(sample_df, figsize=(12, 6))
    plt.title("缺失值可视化（随机抽样）")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "missing_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"缺失值矩阵已保存：{save_path}")


def plot_pollutant_hist(df: pd.DataFrame,
                        pollutant: str = "PM2.5",
                        output_dir: str = "figures") -> None:
    """
    绘制指定污染物的直方图分布，对应 notebook 中的 hist 方法。
    """
    os.makedirs(output_dir, exist_ok=True)

    if pollutant not in df.columns:
        print(f"列 {pollutant} 不在数据集中，跳过直方图绘制。")
        return

    plt.figure(figsize=(8, 4))
    df[pollutant].dropna().plot(kind="hist", bins=50)
    plt.xlabel(pollutant)
    plt.ylabel("Frequency")
    plt.title(f"{pollutant} 分布直方图")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{pollutant}_hist.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"{pollutant} 直方图已保存：{save_path}")


def plot_pm25_time_series(df: pd.DataFrame,
                          station: str = "Aotizhongxin",
                          output_dir: str = "figures") -> None:
    """
    绘制指定站点 PM2.5 的时间序列折线图（按天平均），
    对应 notebook 中常见的折线图 / 时间序列绘制。
    """
    os.makedirs(output_dir, exist_ok=True)

    if "station" not in df.columns:
        print("数据集中没有 station 列，无法按站点筛选。")
        return
    if "datetime" not in df.columns:
        print("数据集中没有 datetime 列，请先调用 create_time_features()。")
        return
    if "PM2.5" not in df.columns:
        print("数据集中没有 PM2.5 列。")
        return

    subset = df[df["station"] == station].copy()
    if subset.empty:
        print(f"未找到站点 {station} 的数据，跳过时间序列绘制。")
        return

    subset = subset.set_index("datetime").sort_index()
    daily_pm25 = subset["PM2.5"].resample("D").mean()

    plt.figure(figsize=(12, 4))
    daily_pm25.plot()
    plt.ylabel("PM2.5 (Daily Mean)")
    plt.title(f"{station} 站点 PM2.5 日均时间序列")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"PM25_timeseries_{station}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PM2.5 时间序列图已保存：{save_path}")


def plot_pm25_box_by_month(df: pd.DataFrame,
                           station: str = "Aotizhongxin",
                           output_dir: str = "figures") -> None:
    """
    绘制指定站点 PM2.5 按月份的箱线图（boxplot），
    对应 notebook 中的箱线图方法。
    """
    os.makedirs(output_dir, exist_ok=True)

    required_cols = {"PM2.5", "month", "station"}
    if not required_cols.issubset(df.columns):
        print(f"缺少列 {required_cols}，无法绘制箱线图。")
        return

    subset = df[df["station"] == station].copy()
    if subset.empty:
        print(f"未找到站点 {station} 的数据，跳过箱线图绘制。")
        return

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="month", y="PM2.5", data=subset)
    plt.xlabel("Month")
    plt.ylabel("PM2.5")
    plt.title(f"{station} 站点 PM2.5 按月份箱线图")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"PM25_boxplot_month_{station}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PM2.5 箱线图已保存：{save_path}")


def plot_correlation_heatmap(df: pd.DataFrame,
                             output_dir: str = "figures") -> None:
    """
    绘制污染物与气象变量之间的相关性热力图，
    对应 notebook 中的 seaborn.heatmap。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 选择一些主要的数值列（可以根据需要调整）
    candidate_cols = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    if len(cols) < 2:
        print("可用于相关性分析的列不足，跳过热力图绘制。")
        return

    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("主要变量相关性热力图")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"相关性热力图已保存：{save_path}")


def main():
    # 这里假设压缩文件与本脚本在同一目录下
    zip_path = "Beijing Multi-Site Air-Quality Data Set.zip"

    # 1. 读取并合并数据
    df = load_beijing_data(zip_path)

    # 2. 构造时间特征
    df = create_time_features(df)

    # 3. 输出基本结构信息
    basic_overview(df)

    # 4. 可视化分析（可按需要注释/取消注释）
    plot_missing_matrix(df, output_dir="figures")
    plot_pollutant_hist(df, pollutant="PM2.5", output_dir="figures")
    plot_pm25_time_series(df, station="Aotizhongxin", output_dir="figures")
    plot_pm25_box_by_month(df, station="Aotizhongxin", output_dir="figures")
    plot_correlation_heatmap(df, output_dir="figures")

    print("\n所有图像已输出到 figures/ 文件夹中，可在报告或 PPT 中使用。")


if __name__ == "__main__":
    main()
