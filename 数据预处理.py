import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ========== 页面导航 ==========
st.title("🧹 数据预处理与统计分析")

uploaded_file = st.file_uploader("📂 上传数据文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # ========== 读取数据 ==========
    # 非缓存模式
    # if uploaded_file.name.endswith(".csv"):
    #     df = pd.read_csv(uploaded_file)
    # else:
    #     df = pd.read_excel(uploaded_file)

    # 数据缓存到 st.session_state，防止每次刷新后数据恢复初始值
    if "df" not in st.session_state:  # 只在第一次上传时加载
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    df = st.session_state.df  # 后续所有操作都基于 session_state.df

    st.success("✅ 数据加载成功！")

    # 功能选择（不放在 sidebar）
    section = st.radio("选择功能模块", ["数据导入与检查",
                                     "缺失值与异常值处理",
                                     "数据变换与特征工程",
                                     "统计分析与可视化"], horizontal=True)

    # ========== 数据导入与检查 ==========
    if section == "数据导入与检查":
        st.subheader("📘 数据导入与检查")
        st.dataframe(df)
        st.write("数据维度：", df.shape)

        col1, col2 = st.columns(2)

        with col1:
            st.write("字段信息：")
            st.write(df.dtypes)

        with col2:
            st.write("缺失值统计：")
            st.write(df.isnull().sum())

    # ========== 缺失值与异常值处理 ==========
    elif section == "缺失值与异常值处理":
        st.subheader("📘 缺失值与异常值处理")

        # 缺失值处理
        st.write("缺失值统计：")
        st.write(df.isnull().sum())
        method = st.selectbox("选择缺失值处理方法",
                              ["不处理", "删除含缺失值的行", "均值填充", "中位数填充", "众数填充"])

        if method != "不处理":
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if method == "删除含缺失值的行":
                        df.dropna(inplace=True)
                    elif method == "均值填充" and df[col].dtype != "object":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "中位数填充" and df[col].dtype != "object":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == "众数填充":
                        df[col].fillna(df[col].mode()[0], inplace=True)
            st.success(f"✅ 已完成 {method}")

        # 显示处理后的数据
        st.dataframe(df)

    # ========== 数据变换与特征工程 ==========
    elif section == "数据变换与特征工程":
        st.subheader("📘 数据变换与特征工程")

        # 数值归一化
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        col = st.selectbox("选择需要归一化的字段", num_cols)
        if st.button("执行 Min-Max 归一化"):
            df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            st.success(f"已生成新字段 {col}_norm")
            st.dataframe(df[[col, col + "_norm"]].head())

        # 类别编码
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            cat_col = st.selectbox("选择需要编码的类别字段", cat_cols)
            if st.button("执行独热编码"):
                df = pd.get_dummies(df, columns=[cat_col], prefix=cat_col)
                st.success("✅ 已完成独热编码")

        # 显示处理后的数据
        st.dataframe(df)

    # ========== 统计分析与可视化 ==========
    elif section == "统计分析与可视化":
        st.subheader("📘 统计分析与可视化")

        st.write("描述性统计：")
        st.write(df.describe())

        # 直方图
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        col = st.selectbox("选择绘制直方图的字段", num_cols)
        if col:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col], bins=20, kde=True, ax=ax, color="#2E86AB", alpha=0.7)
            ax.set_title(f"{col} 分布直方图", fontsize=12, fontweight="bold")
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("频数", fontsize=10)
            st.pyplot(fig)

        # 散点图
        if len(num_cols) >= 2:
            x_col = st.selectbox("选择散点图X轴字段", num_cols, index=0)
            y_col = st.selectbox("选择散点图Y轴字段", num_cols, index=1)
            if x_col and y_col:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color="#A23B72", alpha=0.6)
                ax.set_title(f"{x_col} vs {y_col} 散点图", fontsize=12, fontweight="bold")
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)
                st.pyplot(fig)

        # 相关性热力图
        if st.checkbox("显示相关性热力图"):
            # 计算相关系数矩阵
            corr = df[num_cols].corr()
            # 绘制热力图
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                annot=True,  # 显示相关系数数值
                cmap="coolwarm",  # 颜色映射（红=正相关，蓝=负相关）
                ax=ax,
                fmt=".2f",  # 数值保留2位小数
                linewidths=0.5  # 网格线宽度
            )
            ax.set_title("变量相关性热力图", fontsize=12, fontweight="bold")
            st.pyplot(fig)

    # 导出结果
    st.subheader("📥 导出处理后的数据")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载处理后CSV", csv, "processed_data.csv", "text/csv")
