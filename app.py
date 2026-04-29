import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="结果可视化分析", layout="wide")

# ======================
# 样式
# ======================
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: bold; text-align: center; }
.sub-title { font-size: 1rem; text-align: center; margin-bottom: 1rem; }
.feature-box {
    padding: 6px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    text-align: center;
}
.center-text { text-align:center; margin-top:20px; font-size:20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 模型结果可视化分析</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">选择15个特征值，查看模型结果</div>', unsafe_allow_html=True)

# ======================
# 数据加载
# ======================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "feature_combinations.csv"))
    df_results = pd.read_csv(os.path.join(base_dir, "beta_merged_processed_0418.csv"))
    return df_features, df_results

df_features, df_results = load_data()

# ======================
# 上：特征选择（1行）
# ======================
st.markdown("### 🔧 特征选择")

cols = st.columns(15)
selected_features = {}

for i in range(15):
    with cols[i]:
        st.markdown(f"<div class='feature-box'><b>F{i+1}</b></div>", unsafe_allow_html=True)
        selected_features[f"feature_{i+1}"] = st.radio(
            label=f"feature_{i+1}",
            options=[0, 1],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key=f"f{i+1}"
        )

analyze = st.button("分析模型结果")

# ======================
# 可行性判断函数
# ======================
def is_feasible(f):
    if f["feature_2"] == 1 and f["feature_3"] != 1:
        return False
    if f["feature_4"] == 1 and (f["feature_5"] != 1 or f["feature_6"] != 1):
        return False
    if f["feature_5"] == 1 and f["feature_6"] != 1:
        return False
    if f["feature_7"] == 1 and f["feature_8"] != 1:
        return False
    if f["feature_9"] == 1 and f["feature_10"] != 1:
        return False
    if f["feature_11"] == 1 and (f["feature_12"] != 1 or f["feature_13"] != 1 or f["feature_14"] != 1):
        return False
    if f["feature_12"] == 1 and (f["feature_13"] != 1 or f["feature_14"] != 1):
        return False
    if f["feature_13"] == 1 and f["feature_14"] != 1:
        return False
    return True

# ======================
# 下：仪表盘
# ======================
if analyze:

    feasible = is_feasible(selected_features)

    if not feasible:
        mean_val = 0
        pos_ratio = 0
        neg_ratio = 0
    else:
        match_mask = pd.Series([True] * len(df_features))
        for k, v in selected_features.items():
            match_mask &= (df_features[k] == v)

        matched_ids = df_features.loc[match_mask, "id"].values

        all_results = []
        for mid in matched_ids:
            if mid in df_results["id"].values:
                vals = df_results[df_results["id"] == mid].drop(columns=["id"]).values.flatten()
                all_results.extend(vals)

        all_results = np.array(all_results)

        mean_val = float(np.mean(all_results)) if len(all_results) else 0
        pos_ratio = np.sum(all_results > 0) / len(all_results) if len(all_results) else 0
        neg_ratio = np.sum(all_results < 0) / len(all_results) if len(all_results) else 0

    st.markdown("### 🎯 模型分析仪表盘")

    g1, g2, g3 = st.columns(3)

    # 正占比
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pos_ratio * 100,
            number={"suffix": "%"},
            title={"text": "大于0占比"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # 均值
    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_val,
            number={"valueformat": ".2f"},
            title={"text": "均值"},
            gauge={"axis": {"range": [-1, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # 负占比（新增）
    with g3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=neg_ratio * 100,
            number={"suffix": "%"},
            title={"text": "小于0占比"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 可行性提示（关键）
    # ======================
    if feasible:
        st.markdown(
            "<div class='center-text'><b>Feasible feature combination</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='center-text' style='color:red;'><b>Infeasible feature combination</b></div>",
            unsafe_allow_html=True
        )

st.markdown("---")
