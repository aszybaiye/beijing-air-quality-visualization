import os
import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st

# 基本绘图风格
sns.set(style="whitegrid", font_scale=1.0)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 year, month, day, hour 构造时间相关特征。
    """
    df = df.copy()
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
        df["date"] = df["datetime"].dt.date
        df["weekday"] = df["datetime"].dt.weekday
        df["hour_of_day"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month  # 保证有 month 列
    return df


@st.cache_data(show_spinner=True)
def load_beijing_data_from_zip_bytes(zip_bytes: bytes) -> pd.DataFrame:
    """
    从 zip 字节流读取所有 CSV 文件，并合并。
    用于 Streamlit 的文件上传模式。
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("压缩包中没有找到任何 CSV 文件。")

        dfs = []
        for fname in csv_files:
            with zf.open(fname) as f:
                df = pd.read_csv(f)
                dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = create_time_features(data)
    return data


@st.cache_data(show_spinner=True)
def load_beijing_data_from_path(zip_path: str) -> pd.DataFrame:
    """
    从本地路径读取 zip 文件并加载数据。
    """
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    return load_beijing_data_from_zip_bytes(zip_bytes)


def show_basic_info(df: pd.DataFrame):
    st.subheader("Basic Information / 数据基本信息")
    st.write(f"数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")

    st.write("前 5 行：")
    st.dataframe(df.head())

    st.write("描述性统计：")
    st.dataframe(df.describe())


def show_missing_matrix(df: pd.DataFrame):
    st.subheader("Missing Data Visualization / 缺失值可视化")

    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 4))
    msno.matrix(sample_df, ax=ax)
    st.pyplot(fig)


def show_pollutant_hist(df: pd.DataFrame):
    st.subheader("Pollutant Distribution / 污染物分布直方图")

    candidate_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        st.warning("找不到常见污染物列（PM2.5, PM10, SO2, NO2, CO, O3 等），无法绘制直方图。")
        return

    pollutant = st.selectbox("选择污染物 / Select pollutant", cols, index=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    df[pollutant].dropna().hist(bins=50, ax=ax)
    ax.set_xlabel(pollutant)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{pollutant} histogram")
    st.pyplot(fig)


def show_pm25_time_series_and_boxplot(df: pd.DataFrame):
    if not {"station", "datetime", "PM2.5"}.issubset(df.columns):
        st.warning("缺少 station / datetime / PM2.5 列，无法绘制时间序列和箱线图。")
        return

    st.subheader("PM2.5 Time Series & Monthly Boxplot")

    stations = sorted(df["station"].dropna().unique().tolist())
    if not stations:
        st.warning("当前筛选条件下没有任何站点数据。")
        return

    default_index = stations.index("Aotizhongxin") if "Aotizhongxin" in stations else 0
    station = st.selectbox("选择站点 / Select station", stations, index=default_index)

    subset = df[df["station"] == station].copy()
    subset = subset.dropna(subset=["datetime"])
    if subset.empty:
        st.warning("该站点在当前筛选条件下没有数据。")
        return

    subset = subset.set_index("datetime").sort_index()

    # 时间序列（日均）
    daily_pm25 = subset["PM2.5"].resample("D").mean()

    st.markdown(f"**{station} 站点 PM2.5 日均时间序列**")
    fig, ax = plt.subplots(figsize=(10, 3))
    daily_pm25.plot(ax=ax)
    ax.set_ylabel("PM2.5 (Daily Mean)")
    ax.set_xlabel("")
    st.pyplot(fig)

    # 按月份箱线图
    if "month" not in subset.columns:
        subset["month"] = subset.index.month

    st.markdown(f"**{station} 站点 PM2.5 按月份箱线图**")
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x="month", y="PM2.5", data=subset.reset_index(), ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)


def show_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap / 相关性热力图")

    candidate_cols = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    if len(cols) < 2:
        st.warning("可用于相关性分析的列不足（少于 2 列），无法绘制热力图。")
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    st.pyplot(fig)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    侧边栏筛选：
    - 多选站点
    - 时间范围
    返回筛选后的 DataFrame
    """
    st.sidebar.header("数据筛选 / Filters")
    df = df.copy()

    # 按站点筛选
    if "station" in df.columns:
        stations = sorted(df["station"].dropna().unique().tolist())
        if stations:
            default_stations = stations[:3] if len(stations) > 3 else stations
            selected_stations = st.sidebar.multiselect(
                "选择站点 / Select station(s)",
                stations,
                default=default_stations
            )
            if selected_stations:
                df = df[df["station"].isin(selected_stations)]

    # 按时间范围筛选
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        valid_dt = df["datetime"].dropna()
        if not valid_dt.empty:
            min_date = valid_dt.min().date()
            max_date = valid_dt.max().date()

            date_range = st.sidebar.date_input(
                "选择时间范围 / Date range",
                (min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            # 可能是单个日期或元组，这里统一成 (start, end)
            if isinstance(date_range, (tuple, list)):
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range

            mask = (
                df["datetime"].dt.date >= start_date
            ) & (
                df["datetime"].dt.date <= end_date
            )
            df = df[mask]

    return df


def show_kpi(df: pd.DataFrame):
    """
    显示当前筛选条件下的一些关键指标（KPI）。
    """
    st.subheader("当前筛选条件下的指标 / KPIs")

    col1, col2, col3 = st.columns(3)

    if "PM2.5" in df.columns and not df["PM2.5"].dropna().empty:
        pm25_mean = df["PM2.5"].mean()
        pm25_max = df["PM2.5"].max()
        col1.metric("平均 PM2.5", f"{pm25_mean:.1f}")
        col2.metric("最大 PM2.5", f"{pm25_max:.1f}")
    else:
        col1.metric("平均 PM2.5", "N/A")
        col2.metric("最大 PM2.5", "N/A")

    col3.metric("样本数 / Records", f"{len(df)}")


def main():
    # ===== 顶部显示两个校徽 =====
    logo_col1, logo_col2, _ = st.columns([1, 1, 2])
    with logo_col1:
        if os.path.exists("wut_logo.jpg"):
            st.image("wut_logo.jpg", caption="Wuhan University of Technology", use_container_width=True)
    with logo_col2:
        if os.path.exists("efrei_logo.png"):
            st.image("efrei_logo.png", caption="Efrei Paris", use_container_width=True)

    st.title("Beijing Multi-Site Air Quality Visualization")

    st.markdown(
        """
        本应用基于 **Beijing Multi-Site Air-Quality Data Set**，  
        提供常见的预处理与可视化操作，用于课程《数据可视化》 / 数据分析学习展示。

        - 左侧选择数据来源（本地 ZIP 或上传 ZIP）
        - 可以按站点、时间范围进行筛选
        - 顶部展示当前筛选下的关键指标（KPI）
        - 通过标签页查看概览、分布、时间序列和相关性
        """
    )

    # 1. 加载数据
    st.sidebar.header("数据来源 / Data Source")
    mode = st.sidebar.radio(
        "选择数据加载方式 / Choose how to load data",
        ("使用本地 ZIP 文件（与脚本同目录）", "通过浏览器上传 ZIP 文件"),
    )

    df = None

    if mode == "使用本地 ZIP 文件（与脚本同目录）":
        default_zip = "Beijing Multi-Site Air-Quality Data Set.zip"
        if os.path.exists(default_zip):
            st.sidebar.success(f"找到本地数据文件：{default_zip}")
            try:
                df = load_beijing_data_from_path(default_zip)
            except Exception as e:
                st.error(f"读取本地 ZIP 时出错：{e}")
        else:
            st.sidebar.error(f"当前目录下未找到 {default_zip}，请检查文件名或改用上传方式。")

    else:
        uploaded_file = st.sidebar.file_uploader(
            "上传数据压缩包（zip）/ Upload zip file",
            type=["zip"]
        )
        if uploaded_file is not None:
            try:
                df = load_beijing_data_from_zip_bytes(uploaded_file.read())
                st.sidebar.success("上传并加载数据成功！")
            except Exception as e:
                st.error(f"读取上传的 ZIP 时出错：{e}")

    if df is None:
        st.info("请先在左侧选择数据来源并成功加载数据。")
        st.stop()

    # 2. 应用交互筛选
    df_filtered = apply_filters(df)

    st.markdown(
        f"当前筛选后共有 **{len(df_filtered)}** 条记录。"
    )

    # 3. KPI 指标
    show_kpi(df_filtered)

    if df_filtered.empty:
        st.warning("当前筛选条件下没有数据，请调整筛选条件。")
        st.stop()

    st.markdown("---")

    # 4. Tabs 展示不同分析模块
    tab_overview, tab_distribution, tab_time, tab_corr = st.tabs(
        ["Overview", "Distribution & Missing", "Time Series", "Correlation"]
    )

    with tab_overview:
        st.markdown("### 数据概览 / Data Overview")
        show_basic_info(df_filtered)

    with tab_distribution:
        st.markdown("### 分布与缺失值 / Distribution & Missing")
        if st.checkbox("显示缺失值矩阵 / Show missing data matrix", key="missing_tab"):
            show_missing_matrix(df_filtered)
            st.markdown("---")
        show_pollutant_hist(df_filtered)

    with tab_time:
        st.markdown("### 时间序列与箱线图 / Time Series & Boxplot")
        show_pm25_time_series_and_boxplot(df_filtered)

    with tab_corr:
        st.markdown("### 相关性分析 / Correlation Analysis")
        show_correlation_heatmap(df_filtered)


if __name__ == "__main__":
    main()
