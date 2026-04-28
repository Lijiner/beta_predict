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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== 标题区域 ===== */
.title-wrapper {
    background: linear-gradient(135deg, #0c1220 0%, #1a365d 40%, #0f172a 100%);
    padding: 40px 20px 32px 20px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.35);
    position: relative;
    overflow: hidden;
}

.title-wrapper::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6, #3b82f6);
    background-size: 300% 100%;
    animation: gradientShift 4s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -1px;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

.sub-title {
    font-size: 1.05rem;
    text-align: center;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ===== 特征区 ===== */
.features-section {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    padding: 24px 28px 20px 28px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.features-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 2px solid #f1f5f9;
}

.features-header-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}

/* ===== 按钮 ===== */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8, #3730a3) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}

/* ===== 仪表盘 ===== */
.gauges-container {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 标题
# ======================
st.markdown("""
<div class="title-wrapper">
    <div class="main-title">The Prototyped Decision Support System</div>
    <div class="sub-title">Train Delay Override Analysis & Punctuality Prediction</div>
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
# 特征选择（轻量版）
# ======================
st.markdown('<div class="features-section">', unsafe_allow_html=True)

st.markdown("""
<div class="features-header">
    <div class="features-header-icon">🔧</div>
    <div>
        <div style="font-weight:700;">Feature Configuration</div>
        <div style="font-size:0.8rem;color:#64748b;">select a value for each feature</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 初始化
for i in range(1, 16):
    k = f"feature_{i}"
    if k not in st.session_state:
        st.session_state[k] = 0

def set_feature(i, v):
    st.session_state[f"feature_{i}"] = v

# 3×5 网格
rows = [st.columns(5) for _ in range(3)]

idx = 1
for r in range(3):
    for c in range(5):
        if idx > 15:
            break

        with rows[r][c]:
            val = st.session_state[f"feature_{idx}"]

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-weight:700;
                    color:#1e293b;
                    margin-bottom:6px;
                    padding:4px;
                    border-radius:6px;
                    background:{'#ecfdf5' if val==1 else '#f8fafc'};
                    border:1px solid {'#10b981' if val==1 else '#e2e8f0'};
                ">
                F{idx}
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:
                if st.button("0", key=f"f{idx}_0", use_container_width=True):
                    set_feature(idx, 0)
                    st.rerun()

            with c2:
                if st.button("1", key=f"f{idx}_1", use_container_width=True):
                    set_feature(idx, 1)
                    st.rerun()

        idx += 1

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# START + 组合
# ======================
feat_values = [str(st.session_state[f"feature_{i}"]) for i in range(1, 16)]

st.markdown(f"""
<div style="background:#eff6ff;padding:12px;border-radius:12px;">
Current feature Combination: [{', '.join(feat_values)}]
</div>
""", unsafe_allow_html=True)

analyze = st.button("START", use_container_width=True)

# ======================
# 仪表盘（完整保留原逻辑）
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
        st.error("⚠️ Infeasible Feature Combination")

    else:
        st.success("✅ Feasible Feature Combination")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="gauges-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:60px;color:#94a3b8;">
        📊 Ready to Analyze
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#94a3b8;margin-top:20px;'>© The Prototyped Decision Support System</div>", unsafe_allow_html=True)
