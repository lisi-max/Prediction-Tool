import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor   # 新增
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import time

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 页面设置
st.title("🧑‍💻 模型训练与超参数调优")

# 上传数据
uploaded_file = st.file_uploader("📂 上传数据文件 (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # 如果上传了新文件，清空缓存的数据
    if "df" in st.session_state:
        del st.session_state.df  # 删除缓存中的数据

    # 读取数据并缓存
    if "df" not in st.session_state:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    df = st.session_state.df
    st.success("✅ 数据加载成功！")

    # 显示数据基本信息
    st.subheader("📘 数据概览")
    st.dataframe(df)
    st.write("数据维度：", df.shape)

    # 功能选择
    section = st.radio("选择功能模块", ["数据准备", "模型选择与训练", "模型保存与下载", "模型加载与预测"],
                       horizontal=True)

    # ===================== 数据准备 =====================
    if section == "数据准备":
        st.subheader("📘 数据准备")

        # 选择目标变量
        target_column = st.selectbox("选择目标变量", df.columns)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # 切分训练集和测试集
        test_size = st.slider("选择测试集比例", 0.1, 0.9, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.write(f"训练集大小: {X_train.shape[0]} 行，测试集大小: {X_test.shape[0]} 行")
        st.write("训练集样本：")
        st.dataframe(X_train)
        st.write("测试集样本：")
        st.dataframe(X_test)

        # 缓存训练数据和测试数据，用于跨页面训练
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["feature_dtypes"] = X.dtypes.to_dict()  # 保存每个特征的类型
        st.session_state["target_column"] = target_column

    # ===================== 模型选择与训练 =====================
    elif section == "模型选择与训练":
        st.subheader("📘 模型选择与训练")

        # 获取缓存数据
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]


        # 模型选择
        model_type = st.selectbox("选择训练模型", ["随机森林", "支持向量机", "逻辑回归", "神经网络"])

        if model_type == "随机森林":
            n_estimators = st.slider("选择树的数量", 10, 200, 100)
            max_depth = st.slider("选择树的最大深度", 1, 20, 10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        elif model_type == "支持向量机":
            kernel = st.selectbox("选择核函数", ["linear", "poly", "rbf", "sigmoid"])
            C = st.slider("选择正则化参数 C", 0.1, 10.0, 1.0)
            model = SVC(kernel=kernel, C=C)

        elif model_type == "逻辑回归":
            C = st.slider("选择正则化参数 C", 0.01, 10.0, 1.0)
            model = LogisticRegression(C=C)

        elif model_type == "神经网络":
            hidden_layer_sizes = st.text_input("输入隐藏层结构 (例如: 100,50 表示两层)", "100,50")
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden_layer_sizes.split(","))
            activation = st.selectbox("选择激活函数", ["relu", "tanh", "logistic"])
            max_iter = st.slider("最大迭代次数", 100, 2000, 500)
            # 优化器选择
            solver = st.selectbox(
                "选择优化器（solver）",
                ["adam", "sgd", "lbfgs"],  # 常见的三种优化算法
            )
            # 学习率输入（浮点数）
            learning_rate_init = st.number_input(
                label="学习率",  # 输入框标签
                min_value=0.0001,  # 下限
                max_value=0.1,  # 上限
                value=0.001,  # 默认值
                step=0.0001,  # 每次增减的步长
                format="%.4f"  # 显示格式（保留4位小数，确保0.0001这类小数值正常显示）
            )
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,  # 两层：100 和 50 个神经元
                activation=activation,  # 激活函数 relu/tanh/logistic
                solver=solver,  # 优化器 adam
                learning_rate_init=learning_rate_init,  # 学习率
                max_iter=max_iter,  # 最大迭代次数
                random_state=42
            )

        # 训练按钮
        if st.button("开始训练"):
            status_text = st.empty()
            status_text.text("🚀 模型训练中，请稍候...")

            model.fit(X_train, y_train)
            st.success(f"✅ 模型训练完成！")

            # 训练集评估
            y_train_pred = model.predict(X_train)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_r2 = r2_score(y_train, y_train_pred)

            # 测试集评估
            y_test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2 = r2_score(y_test, y_test_pred)

            # 构建评估结果表格
            eval_df = pd.DataFrame({
                "数据集": ["训练集", "测试集"],
                "MAE": [train_mae, test_mae],
                "RMSE": [train_rmse, test_rmse],
                "R²": [train_r2, test_r2]
            })

            # 用表格展示
            st.write("### 模型评估结果")
            st.dataframe(eval_df, use_container_width=True)  # 可以滚动、支持排序

            # 缓存模型训练结果，用于跨页面评估和保存
            st.session_state["model"] = model
            st.session_state["model_type"] = model_type

    # ===================== 模型保存与下载 =====================
    elif section == "模型保存与下载":
        st.subheader("📘 模型保存与下载")
        # 检查模型是否已训练
        if "model" not in st.session_state:
            st.warning("请先进行模型训练，训练完成后才能保存和下载模型。")
        else:
            # 获取缓存数据
            model = st.session_state["model"]
            model_type = st.session_state["model_type"]

            # 直接使用简短的文件名
            model_filename = f"{model_type}_model.pkl"

            # 使用joblib将模型序列化为字节流
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)  # 移动到字节流的开始位置

            # 提供文件下载按钮
            st.download_button(
                label="下载训练好的模型",
                data=model_bytes,
                file_name=model_filename,
                mime="application/octet-stream"
            )

    # ===================== 模型评估与结果 =====================
    elif section == "模型加载与预测":
        st.subheader("📘 模型加载与预测")

        # 检查是否有列名
        if "feature_columns" not in st.session_state:
            st.warning("请先上传数据并训练模型，列名信息才能被保存用于预测。")
        else:
            # 从文件上传加载模型
            uploaded_model = st.file_uploader("上传训练好的模型 (pkl)", type=["pkl"])

            if uploaded_model:
                # 加载模型
                model = joblib.load(uploaded_model)
                st.success("✅ 模型加载成功！")

                # 获取保存的特征列名和数据类型
                feature_columns = st.session_state["feature_columns"]
                feature_dtypes = st.session_state["feature_dtypes"]

                # 获取保存的目标列名
                target_column = st.session_state["target_column"]

                # 让用户输入新的数据进行预测
                st.subheader("请输入新的数据进行预测")

                # 输入特征值
                features_input = {}
                for col in feature_columns:
                    dtype = feature_dtypes[col]

                    if dtype == 'int64' or dtype == 'float64':  # 数值型特征
                        features_input[col] = st.number_input(f"请输入{col}的值", value=0.0)
                    elif dtype == 'object':  # 类别型特征
                        unique_values = st.session_state.df[col].unique()  # 获取该特征的所有唯一值
                        features_input[col] = st.selectbox(f"选择{col}的值", options=unique_values)
                    elif dtype == 'bool':  # 布尔型特征
                        features_input[col] = st.checkbox(f"选择{col}", value=False)
                    else:  # 如果是其他类型
                        features_input[col] = st.text_input(f"请输入{col}的值")

                # 将输入的数据转化为DataFrame
                input_data = pd.DataFrame([features_input])

                # 使用模型进行预测
                if st.button("开始预测"):
                    prediction = model.predict(input_data)
                    st.write(f"预测的 {target_column} 值为: {prediction[0]:.2f}")