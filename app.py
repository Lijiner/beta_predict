import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="The Prototyped Decision Support System", layout="wide")

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
.center-text { text-align:center; margin-top:10px; font-size:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">The Prototyped Decision Support System</div>', unsafe_allow_html=True)
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
# 特征选择
# ======================
st.markdown("### Select a value for each feature")

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

# 当前组合
feat_values = [str(selected_features[f"feature_{i}"]) for i in range(1, 16)]
st.markdown(
    f"""<div style="background:#f3f4f6; padding:8px; border-radius:6px; margin-top:10px;">
    Current feature combination: <b>[{', '.join(feat_values)}]</b>
    </div>""",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)  # 向下推
analyze = st.button("START")

# ======================
# 规则校验
# ======================
def is_feasible(f):
    if f["feature_2"] == 1 and f["feature_3"] != 1: return False
    if f["feature_4"] == 1 and (f["feature_5"] != 1 or f["feature_6"] != 1): return False
    if f["feature_5"] == 1 and f["feature_6"] != 1: return False
    if f["feature_7"] == 1 and f["feature_8"] != 1: return False
    if f["feature_9"] == 1 and f["feature_10"] != 1: return False
    if f["feature_11"] == 1 and (f["feature_12"] != 1 or f["feature_13"] != 1 or f["feature_14"] != 1): return False
    if f["feature_12"] == 1 and (f["feature_13"] != 1 or f["feature_14"] != 1): return False
    if f["feature_13"] == 1 and f["feature_14"] != 1: return False
    return True

# ======================
# 仪表盘
# ======================
if analyze:

    feasible = is_feasible(selected_features)

    if feasible:
        match_mask = pd.Series([True] * len(df_features))
        for k, v in selected_features.items():
            match_mask &= (df_features[k] == v)

        df_merge = df_features[match_mask].merge(df_results, on="id", how="inner")

        if len(df_merge) == 0:
            st.error("未匹配到数据")
            all_results = np.array([])
        else:
            all_results = df_merge.drop(columns=["id"]).values.flatten()
    else:
        all_results = np.array([])

    # ======================
    # 计算指标
    # ======================
    if len(all_results) > 0:
        mean_val = float(np.mean(all_results))
        pos_ratio = np.sum(all_results > 0) / len(all_results)
        neg_ratio = np.sum(all_results < 0) / len(all_results)

        # ✅ 动态范围（核心）
        data_min = float(np.min(all_results))
        data_max = float(np.max(all_results))

        gauge_min = min(0, data_min * 1.1)
        gauge_max = max(100, data_max * 1.1)

    else:
        mean_val = 0
        pos_ratio = 0
        neg_ratio = 0
        gauge_min, gauge_max = 0, 100

    g1, g2, g3 = st.columns(3)

    # ======================
    # 1️⃣ 大于0（红）
    # ======================
    with g1:
        st.markdown(
            "<div style='text-align:center; font-weight:700; font-size:20px;'>"
            "Predicted probability that override harms train punctuality"
            "</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pos_ratio * 100,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#dc2626"}}
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 2️⃣ 均值（蓝）🔥动态范围
    # ======================
    with g2:
        st.markdown(
            "<div style='text-align:center; font-weight:700; font-size:20px;'>"
            "Per train section dredicted change in train delay if overriden"
            "</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_val,
            number={
                "suffix":" seconds",
                "font": {"size": 30}
            },
            gauge={
                "axis": {"range": [gauge_min, gauge_max]},
                "bar": {"color": "#2563eb"}
            }
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 3️⃣ 小于0（绿）
    # ======================
    with g3:
        st.markdown(
            "<div style='text-align:center; font-weight:700; font-size:20px;'>"
            "Predicted probability that override improves train punctuality"
            "</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=neg_ratio * 100,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#059669"}}
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 可行性提示
    # ======================
    if feasible:
        st.markdown(
            "<div style='text-align:center; font-size:28px;'><b>Feasible feature combination</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center; color:red; font-size:28px;'><b>Infeasible feature combination</b></div>",
            unsafe_allow_html=True
        )

st.markdown("---")
