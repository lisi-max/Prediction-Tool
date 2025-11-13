import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import uuid

# ========== 页面导航 ==========
st.title("🏠 房价预测系统 - 数据采集")
page = st.radio("请选择数据采集方式：", ["模拟生成", "文件上传", "爬虫采集", "数据库读取"], horizontal=True)

data = None  # 存放最终的数据

# ========== 模拟生成数据 ==========
if page == "模拟生成":
    st.subheader("🔹 模拟生成房价数据")
    n = st.slider("数据量", 50, 500, 100)
    np.random.seed(42)

    area = np.random.randint(30, 150, n)        # 面积
    floor = np.random.randint(1, 30, n)         # 楼层
    age = np.random.randint(0, 30, n)           # 房龄
    distance_to_metro = np.random.randint(100, 5000, n)  # 距离地铁（米）
    decoration = np.random.choice(["简装", "精装", "豪装"], n)  # 装修情况

    # 模拟价格（非线性组合 + 噪声）
    price = (area * 2000
             + floor * 800
             - age * 500
             - distance_to_metro * 5
             + np.where(decoration=="简装", 0, np.where(decoration=="豪装", 40000, 20000))
             + np.random.randint(-20000, 20000, n))

    # 组装 DataFrame
    data = pd.DataFrame({
        "面积(平米)": area,
        "楼层": floor,
        "房龄(年)": age,
        "距地铁(m)": distance_to_metro,
        "装修": decoration,
        "价格(元)": price
    })

    st.dataframe(data)

# ========== 文件上传 ==========
elif page == "文件上传":
    st.subheader("🔹 上传房价数据文件 (CSV/Excel)")
    uploaded_file = st.file_uploader("上传文件", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.success("文件上传成功！")
        st.dataframe(data)

# ========== 爬虫采集（示例） ==========
elif page == "爬虫采集":
    import requests
    from lxml import etree
    import pandas as pd

    st.subheader("🔹 爬虫采集房价数据示例 (requests + XPath)")

    # URL（深圳二手房）
    url = "https://sz.lianjia.com/ershoufang/pg1/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        tree = etree.HTML(response.text)

        # 房屋标题
        titles = tree.xpath('//div[@class="title"]/a/text()')

        # 房屋位置
        region = tree.xpath('//div[@class="positionInfo"]/a[@data-el="region"]/text()')
        # region = ['-'.join(a.xpath('.//a/text()')) for a in tree.xpath('//div[@class="positionInfo"]')]  # 位置全称

        # 房屋信息
        houseInfo = tree.xpath('//div[@class="houseInfo"]/text()')

        # 房屋户型
        layouts = [info.split('|')[0].strip() for info in houseInfo]

        # 房屋面积
        areas = [info.split('|')[1].strip()[:-2] for info in houseInfo]

        # 房屋总价
        prices = tree.xpath('//div[@class="priceInfo"]/div[@class="totalPrice totalPrice2"]/span/text()')

        # 房屋单价
        unit_prices = tree.xpath('//div[@class="priceInfo"]/div[@class="unitPrice"]/span/text()')
        unit_prices = [unit_price[:-3] for unit_price in unit_prices]

        # 组装 DataFrame
        data = pd.DataFrame({
            "房源标题": titles,
            "位置": region,
            "户型": layouts,
            "面积（平米）": areas,
            "总价(万)": prices,
            "单价(元/平米)": unit_prices
        })

        st.dataframe(data)

    else:
        st.error(f"请求失败，状态码: {response.status_code}")


# ========== 数据库读取 ==========
elif page == "数据库读取":
    st.subheader("🔹 从数据库读取数据 (SQLite 文件)")

    # 上传 SQLite 数据库文件
    uploaded_db = st.file_uploader("请选择 SQLite 数据库文件（.db）", type=["db"])

    if uploaded_db is not None:
        try:
            # 使用 in-memory 方式读取上传的数据库
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_db.read())
                db_path = tmp_file.name

            # 连接 SQLite 数据库
            conn = sqlite3.connect(db_path)
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql(query, conn)

            if len(tables) == 0:
                st.warning("数据库中没有表！")
            else:
                # 默认读取第一个表
                table_name = tables['name'][0]
                data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                st.dataframe(data)

            conn.close()

        except Exception as e:
            st.error(f"❌ 读取数据库出错：{e}")
    else:
        st.info("请上传 SQLite 数据库文件以读取数据。")

# ========== 通用功能 ==========
if data is not None:
    csv = data.to_csv(index=False).encode("utf-8-sig")

    # 设置两列布局
    col1, col2 = st.columns(2)

    with col1:
        # 为每个用户生成唯一的文件名
        unique_filename = f"housing_{uuid.uuid4().hex}.db"

        # 直接生成 SQLite 数据库文件
        conn = sqlite3.connect(unique_filename)  # 使用唯一的文件名
        data.to_sql("housing", conn, index=False, if_exists="replace")
        conn.close()

        # 读取数据库文件内容
        with open(unique_filename, "rb") as f:
            db_bytes = f.read()

        # 提供数据库下载按钮
        st.download_button(
            label="💾 下载数据库 (SQLite)",
            data=db_bytes,
            file_name=unique_filename,
            mime="application/octet-stream"
        )

        # 下载后自动删除临时数据库文件
        os.remove(unique_filename)
        # print(f"已删除临时文件: {unique_filename}")  # 提示文件已删除

    with col2:
        st.download_button("📥 下载数据CSV", csv, "data.csv", "text/csv")
