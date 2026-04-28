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
# 样式（原样保留）
# ======================
st.markdown("""
<style>
/* 这里你的CSS完全不动（已省略，原样保留即可） */
</style>
""", unsafe_allow_html=True)

# ======================
# 标题区域
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
# 特征选择（已改：极简版，无卡片）
# ======================
st.markdown('<div class="features-section">', unsafe_allow_html=True)

st.markdown("""
<div class="features-header">
    <div class="features-header-icon">🔧</div>
    <div>
        <div class="features-header-text">Feature Configuration</div>
        <div class="features-header-desc">select a value for each feature</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 初始化特征值
for i in range(1, 16):
    key = f"feature_{i}"
    if key not in st.session_state:
        st.session_state[key] = 0

def set_feature(feat_num, value):
    st.session_state[f"feature_{feat_num}"] = value

# ===== 3×5 极简布局 =====
rows = [st.columns(5) for _ in range(3)]

feature_idx = 1
for r in range(3):
    for c in range(5):
        if feature_idx > 15:
            break

        with rows[r][c]:
            val = st.session_state[f"feature_{feature_idx}"]

            # 仅保留编号（无卡片）
            st.markdown(
                f"<div style='text-align:center; font-weight:700; color:#1e293b; margin-bottom:6px;'>F{feature_idx}</div>",
                unsafe_allow_html=True
            )

            # 0 / 1 按钮
            c1, c2 = st.columns(2)

            with c1:
                if st.button("0", key=f"btn_f{feature_idx}_0", use_container_width=True):
                    set_feature(feature_idx, 0)
                    st.rerun()

            with c2:
                if st.button("1", key=f"btn_f{feature_idx}_1", use_container_width=True):
                    set_feature(feature_idx, 1)
                    st.rerun()

        feature_idx += 1

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 组合显示 + START 按钮
# ======================
feat_values = [str(st.session_state[f"feature_{i}"]) for i in range(1, 16)]

st.markdown(
    f"""<div class="action-bar">
        <div class="combo-display">
            <span class="combo-label">Current feature Combination</span>
            <span class="combo-value">[{', '.join(feat_values)}]</span>
        </div>
    </div>""",
    unsafe_allow_html=True
)

btn_col1, btn_col2, btn_col3 = st.columns([1, 0.35, 1])
with btn_col2:
    analyze = st.button("START", use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ======================
# 仪表盘区域（完全不动）
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
        st.error("Infeasible Feature Combination")
    else:
        st.success("Feasible Feature Combination")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="gauges-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#94a3b8;">
        <div style="font-size:3rem; margin-bottom:16px;">📊</div>
        <div style="font-size:1.2rem; font-weight:600; color:#475569; margin-bottom:8px;">Ready to Analyze</div>
        <div style="font-size:0.95rem;">Select feature values above and click <b>START</b> to view results</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='footer-text'>© The Prototyped Decision Support System</div>", unsafe_allow_html=True)
