import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ======================
# 页面配置
# ======================
st.set_page_config(
    page_title="The Prototyped Decision Support System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================
# 极简样式
# ======================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

/* 标题 */
.title-wrapper {
    padding: 20px 0;
    text-align: center;
}

.main-title {
    font-size: 28px;
    font-weight: 700;
}

.sub-title {
    font-size: 14px;
    color: #666;
    margin-top: 6px;
}

/* 特征区域 */
.features-section {
    padding: 10px 0;
}

.features-header-text {
    font-weight: 600;
    font-size: 16px;
}

.features-header-desc {
    font-size: 12px;
    color: #888;
}

/* 15列 */
.features-row {
    display: flex;
    gap: 6px;
}

/* 特征块（极简） */
.feature-unit {
    flex: 1;
    text-align: center;
    padding: 6px 2px;
    border: 1px solid #ddd;
}

/* 统一无颜色 */
.feature-name {
    font-size: 10px;
    color: #333;
    margin-bottom: 4px;
}

.feature-value {
    font-size: 14px;
    font-weight: 600;
    color: #000;   /* 不区分0/1颜色 */
}

/* 按钮极简 */
.stButton > button {
    width: 100%;
    background: #f5f5f5 !important;
    color: #000 !important;
    border: 1px solid #ddd !important;
    font-size: 12px !important;
    padding: 2px !important;
}

/* 组合栏 */
.action-bar {
    margin-top: 10px;
    padding: 10px;
    border: 1px solid #eee;
}

.combo-value {
    font-family: monospace;
    font-size: 12px;
}

/* 仪表盘 */
.gauges-container {
    margin-top: 20px;
}

.gauge-card {
    border: 1px solid #ddd;
    padding: 10px;
}

/* footer */
.footer-text {
    text-align: center;
    font-size: 12px;
    color: #999;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 标题
# ======================
st.markdown("""
<div class="title-wrapper">
    <div class="main-title">The Prototyped Decision Support System</div>
    <div class="sub-title">Train Delay Override Analysis &amp; Punctuality Prediction</div>
</div>
""", unsafe_allow_html=True)

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
# 初始化特征
# ======================
for i in range(1, 16):
    key = f"feature_{i}"
    if key not in st.session_state:
        st.session_state[key] = 0

def set_feature(i, v):
    st.session_state[f"feature_{i}"] = v

# ======================
# 特征选择（一行15个）
# ======================
st.markdown('<div class="features-section">', unsafe_allow_html=True)

cols = st.columns(15)

for i in range(1, 16):
    with cols[i-1]:
        val = st.session_state[f"feature_{i}"]

        st.markdown(f"""
        <div class="feature-unit">
            <div class="feature-name">F{i}</div>
            <div class="feature-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

        st.button("0", key=f"f{i}_0", on_click=set_feature, args=(i, 0))
        st.button("1", key=f"f{i}_1", on_click=set_feature, args=(i, 1))

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 组合显示
# ======================
feat_values = [str(st.session_state[f"feature_{i}"]) for i in range(1, 16)]

st.markdown(f"""
<div class="action-bar">
    Current feature Combination:
    <div class="combo-value">[{', '.join(feat_values)}]</div>
</div>
""", unsafe_allow_html=True)

analyze = st.button("START")

# ======================
# 结果区域
# ======================
if analyze:

    selected_features = {f"feature_{i}": st.session_state[f"feature_{i}"] for i in range(1, 16)}

    f = selected_features
    infeasible = False

    if f["feature_2"] == 1 and f["feature_3"] != 1:
        infeasible = True
    if f["feature_4"] == 1 and (f["feature_5"] == 1 or f["feature_6"] != 1):
        infeasible = True
    if f["feature_5"] == 1 and f["feature_6"] != 1:
        infeasible = True
    if f["feature_7"] == 1 and f["feature_8"] != 1:
        infeasible = True
    if f["feature_9"] == 1 and f["feature_10"] != 1:
        infeasible = True
    if f["feature_11"] == 1 and (f["feature_12"] == 1 or f["feature_13"] != 1 or f["feature_14"] == 1):
        infeasible = True
    if f["feature_12"] == 1 and (f["feature_13"] == 1 or f["feature_14"] != 1):
        infeasible = True
    if f["feature_13"] == 1 and f["feature_14"] != 1:
        infeasible = True

    st.markdown('<div class="gauges-container">', unsafe_allow_html=True)

    if infeasible:

        st.markdown("""
        <div class="gauge-card">
            ⚠️ Infeasible Feature Combination
        </div>
        """, unsafe_allow_html=True)

    else:

        match_mask = pd.Series([True] * len(df_features))
        for k, v in selected_features.items():
            match_mask &= (df_features[k] == v)

        matched_ids = df_features.loc[match_mask, "id"].values

        all_results = []
        for mid in matched_ids:
            if mid in df_results["id"].values:
                row = df_results.loc[df_results["id"] == mid].drop(columns=["id"]).values.flatten()
                all_results.extend(row)

        if len(all_results) > 0:
            all_results = np.array(all_results)

            mean_val = np.mean(all_results)
            pos_ratio = np.sum(all_results > 0) / len(all_results)
            neg_ratio = np.sum(all_results < 0) / len(all_results)

            st.markdown(f"""
            <div class="gauge-card">
                Mean: {mean_val}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="gauge-card">
                Harm: {pos_ratio * 100}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="gauge-card">
                Improve: {neg_ratio * 100}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center; margin-top:10px;">
                ✅ Feasible Feature Combination
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#888;">
        Ready to Analyze
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer-text'>© The Prototyped Decision Support System</div>", unsafe_allow_html=True)
