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
# 样式 (保持紧凑风格)
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.title-wrapper { background: linear-gradient(135deg, #0c1220 0%, #1a365d 40%, #0f172a 100%); padding: 30px 20px; border-radius: 16px; margin-bottom: 20px; text-align: center; color: white; }
.main-title { font-size: 2.2rem; font-weight: 900; margin-bottom: 5px; }
.sub-title { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.feature-panel { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 4px; text-align: center; margin-bottom: 8px; }
.feature-name { font-size: 0.7rem; font-weight: 700; color: #475569; margin-bottom: 4px; }
.feature-val { font-size: 1.1rem; font-weight: 800; padding: 4px; border-radius: 6px; margin-bottom: 4px; }
.val-on { background: #059669; color: white; }
.val-off { background: #e2e8f0; color: #64748b; }

.stButton > button { padding: 4px 8px !important; font-size: 0.75rem !important; min-height: 28px !important; }
.gauges-container { background: #ffffff; border-radius: 16px; padding: 20px; border: 1px solid #e2e8f0; }
.result-banner { text-align: center; font-weight: 800; padding: 15px; border-radius: 10px; margin-top: 15px; }
.result-feasible { background: #ecfdf5; color: #047857; border: 1px solid #6ee7b7; }
.result-infeasible { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
</style>
""", unsafe_allow_html=True)

# ======================
# 标题与加载
# ======================
st.markdown("""
<div class="title-wrapper">
    <div class="main-title">The Prototyped Decision Support System</div>
    <div class="sub-title">Train Delay Override Analysis & Punctuality Prediction</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "feature_combinations.csv"))
    df_results = pd.read_csv(os.path.join(base_dir, "beta_merged_processed_0418.csv"))
    return df_features, df_results

df_features, df_results = load_data()

# ======================
# 特征选择 (紧凑的一行)
# ======================
for i in range(1, 16):
    if f"feature_{i}" not in st.session_state:
        st.session_state[f"feature_{i}"] = 0

cols = st.columns(15)
for i in range(1, 16):
    with cols[i-1]:
        val = st.session_state[f"feature_{i}"]
        st.markdown(f"""
        <div class="feature-panel">
            <div class="feature-name">F{i}</div>
            <div class="feature-val {'val-on' if val==1 else 'val-off'}">{val}</div>
        </div>
        """, unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("0", key=f"btn_{i}_0"):
                st.session_state[f"feature_{i}"] = 0
                st.rerun()
        with b2:
            if st.button("1", key=f"btn_{i}_1"):
                st.session_state[f"feature_{i}"] = 1
                st.rerun()

st.divider()

# ======================
# 逻辑处理与仪表盘
# ======================
selected_features = {f"feature_{i}": st.session_state[f"feature_{i}"] for i in range(1, 16)}
analyze = st.button("START ANALYSIS", use_container_width=True)

if analyze:
    # 逻辑校验
    f = selected_features
    infeasible = (
        (f["feature_2"] == 1 and f["feature_3"] != 1) or
        (f["feature_4"] == 1 and (f["feature_5"] == 1 or f["feature_6"] != 1)) or
        (f["feature_5"] == 1 and f["feature_6"] != 1) or
        (f["feature_7"] == 1 and f["feature_8"] != 1) or
        (f["feature_9"] == 1 and f["feature_10"] != 1) or
        (f["feature_11"] == 1 and (f["feature_12"] == 1 or f["feature_13"] != 1 or f["feature_14"] == 1)) or
        (f["feature_12"] == 1 and (f["feature_13"] == 1 or f["feature_14"] != 1)) or
        (f["feature_13"] == 1 and f["feature_14"] != 1)
    )

    st.markdown('<div class="gauges-container">', unsafe_allow_html=True)
    if infeasible:
        st.markdown('<div class="result-banner result-infeasible">⚠️ Infeasible Feature Combination</div>', unsafe_allow_html=True)
    else:
        match_mask = pd.Series([True] * len(df_features))
        for feat_name, feat_val in selected_features.items():
            match_mask = match_mask & (df_features[feat_name] == feat_val)
        
        matched_ids = df_features.loc[match_mask, "id"].values
        if len(matched_ids) == 0:
            st.error("No matches found.")
        else:
            all_results = []
            for mid in matched_ids:
                if mid in df_results["id"].values:
                    all_results.extend(df_results.loc[df_results["id"] == mid].drop(columns=["id"]).values.flatten())
            
            if all_results:
                arr = np.array(all_results)
                m, p, n = np.mean(arr), np.mean(arr > 0)*100, np.mean(arr < 0)*100
                
                cols = st.columns(3)
                for c, title, val, sfx in zip(cols, ["Mean Delay", "Harm Prob", "Improve Prob"], [m, p, n], ["s", "%", "%"]):
                    with c:
                        fig = go.Figure(go.Indicator(mode="gauge+number", value=val, number={"suffix": sfx}, gauge={"axis": {"range": [None, None]}}))
                        fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"<div style='text-align:center;'><b>{title}</b></div>", unsafe_allow_html=True)
                st.markdown('<div class="result-banner result-feasible">✅ Feasible Feature Combination</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
