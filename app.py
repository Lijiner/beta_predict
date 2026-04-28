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
# 样式 (保持不变)
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

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

.main-title { font-size: 2.8rem; font-weight: 900; text-align: center; color: #ffffff; margin-bottom: 8px; letter-spacing: -1px; text-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.sub-title { font-size: 1.05rem; text-align: center; color: #94a3b8; font-weight: 400; letter-spacing: 1px; text-transform: uppercase; }

.features-section {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    padding: 24px 28px 20px 28px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

.features-header { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 2px solid #f1f5f9; }
.features-header-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #1d4ed8); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: white; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
.features-header-text { font-size: 1.15rem; font-weight: 700; color: #1e293b; }
.features-header-desc { font-size: 0.8rem; color: #64748b; margin-top: 1px; }

.feature-unit {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 4px 8px 4px;
    text-align: center;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 10px;
}
.feature-unit-selected { border-color: #059669; background: #ecfdf5; }

.feature-name { font-size: 0.7rem; font-weight: 700; color: #475569; margin-bottom: 6px; letter-spacing: 0.5px; }
.feature-value { width: 32px; height: 32px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; margin: 0 auto; }
.feature-value-off { background: #f1f5f9; color: #94a3b8; border: 2px solid #e2e8f0; }
.feature-value-on { background: linear-gradient(135deg, #059669, #10b981); color: #ffffff; border: 2px solid #059669; box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3); }
.feature-indicator { width: 6px; height: 6px; border-radius: 50%; margin: 6px auto 0 auto; }
.indicator-off { background: #cbd5e1; }
.indicator-on { background: #10b981; box-shadow: 0 0 6px rgba(16, 185, 129, 0.5); }

.action-bar { background: linear-gradient(135deg, #eff6ff, #f0f9ff); border: 1.5px solid #bfdbfe; border-radius: 16px; padding: 14px 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.combo-display { display: flex; align-items: center; gap: 10px; flex: 1; }
.combo-label { color: #475569; font-weight: 600; font-size: 0.85rem; white-space: nowrap; }
.combo-value { font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 700; color: #1e40af; font-size: 0.9rem; background: #ffffff; padding: 6px 14px; border-radius: 10px; border: 1.5px solid #bfdbfe; box-shadow: 0 1px 4px rgba(0,0,0,0.04); letter-spacing: 0.5px; }

.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8, #3730a3) !important; color: white !important; font-weight: 700 !important; font-size: 1rem !important; padding: 10px 28px !important; border-radius: 12px !important; border: none !important; box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important; transition: all 0.3s ease !important; letter-spacing: 0.5px; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(37, 99, 235, 0.5) !important; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, #cbd5e1, transparent); margin: 20px 0; }
.gauges-container { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); border-radius: 20px; padding: 28px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
.gauge-card { background: #ffffff; border-radius: 16px; padding: 20px 16px 12px 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 12px rgba(0,0,0,0.05); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
.gauge-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.12); }
.gauge-header { text-align: center; margin-bottom: 12px; }
.gauge-icon { width: 44px; height: 44px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-bottom: 8px; }
.gauge-title-text { font-size: 1rem; font-weight: 700; color: #1e293b; }
.gauge-subtitle { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
.result-banner { text-align: center; font-weight: 800; font-size: 1.25rem; padding: 18px 24px; border-radius: 14px; margin-top: 20px; display: flex; align-items: center; justify-content: center; gap: 10px; }
.result-feasible { background: linear-gradient(135deg, #ecfdf5, #d1fae5); color: #047857; border: 2px solid #6ee7b7; }
.result-infeasible { background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #b91c1c; border: 2px solid #fca5a5; }
.footer-text { text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 24px; margin-top: 16px; }
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
# 特征选择面板 (已优化为3x5网格)
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

# 初始化
for i in range(1, 16):
    if f"feature_{i}" not in st.session_state:
        st.session_state[f"feature_{i}"] = 0

def set_feature(feat_num, value):
    st.session_state[f"feature_{feat_num}"] = value

# 以 5 列为一行进行布局
for row in range(0, 15, 5):
    cols = st.columns(5)
    for col_idx in range(5):
        feat_idx = row + col_idx + 1
        if feat_idx <= 15:
            with cols[col_idx]:
                val = st.session_state[f"feature_{feat_idx}"]
                card_class = "feature-unit feature-unit-selected" if val == 1 else "feature-unit"
                indicator_class = "feature-indicator indicator-on" if val == 1 else "feature-indicator indicator-off"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="feature-name">F{feat_idx}</div>
                    <div class="feature-value {'feature-value-on' if val == 1 else 'feature-value-off'}">{val}</div>
                    <div class="{indicator_class}"></div>
                </div>
                """, unsafe_allow_html=True)
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("0", key=f"btn_f{feat_idx}_0", use_container_width=True, type="primary" if val == 0 else "secondary"):
                        set_feature(feat_idx, 0)
                        st.rerun()
                with b2:
                    if st.button("1", key=f"btn_f{feat_idx}_1", use_container_width=True, type="primary" if val == 1 else "secondary"):
                        set_feature(feat_idx, 1)
                        st.rerun()

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
# 仪表盘区域 (逻辑保持)
# ======================
if analyze:
    selected_features = {f"feature_{i}": st.session_state[f"feature_{i}"] for i in range(1, 16)}
    f = selected_features
    infeasible = False
    
    # 逻辑校验规则
    if f["feature_2"] == 1 and f["feature_3"] != 1: infeasible = True
    if f["feature_4"] == 1 and (f["feature_5"] == 1 or f["feature_6"] != 1): infeasible = True
    if f["feature_5"] == 1 and f["feature_6"] != 1: infeasible = True
    if f["feature_7"] == 1 and f["feature_8"] != 1: infeasible = True
    if f["feature_9"] == 1 and f["feature_10"] != 1: infeasible = True
    if f["feature_11"] == 1 and (f["feature_12"] == 1 or f["feature_13"] != 1 or f["feature_14"] == 1): infeasible = True
    if f["feature_12"] == 1 and (f["feature_13"] == 1 or f["feature_14"] != 1): infeasible = True
    if f["feature_13"] == 1 and f["feature_14"] != 1: infeasible = True

    st.markdown('<div class="gauges-container">', unsafe_allow_html=True)
    if infeasible:
        st.markdown('<div class="result-banner result-infeasible">⚠️ Infeasible Feature Combination</div>', unsafe_allow_html=True)
    else:
        # 数据处理
        match_mask = pd.Series([True] * len(df_features))
        for feat_name, feat_val in selected_features.items():
            match_mask = match_mask & (df_features[feat_name] == feat_val)
        matched_ids = df_features.loc[match_mask, "id"].values
        
        if len(matched_ids) == 0:
            st.error("❌ 未找到完全匹配的特征组合")
        else:
            all_results = []
            for mid in matched_ids:
                if mid in df_results["id"].values:
                    row_vals = df_results.loc[df_results["id"] == mid].drop(columns=["id"]).values.flatten()
                    all_results.extend(row_vals)
            
            if len(all_results) > 0:
                all_results = np.array(all_results)
                mean_val = float(np.mean(all_results))
                pos_ratio = np.sum(all_results > 0) / len(all_results)
                neg_ratio = np.sum(all_results < 0) / len(all_results)
                
                # 绘制仪表盘
                cols = st.columns(3)
                metrics = [
                    ("Mean", f"{mean_val:.2f} s", "#2563eb", "📊"),
                    ("Harm", f"{pos_ratio*100:.1f} %", "#059669", "🌟"),
                    ("Improve", f"{neg_ratio*100:.1f} %", "#dc2626", "💥")
                ]
                for i, col in enumerate(cols):
                    with col:
                        st.markdown(f"""
                        <div class="gauge-card">
                            <div class="gauge-header">
                                <div class="gauge-icon" style="background:#f1f5f9;">{metrics[i][3]}</div>
                                <div class="gauge-title-text">{metrics[i][0]}</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color:{metrics[i][2]}; margin-top:10px;">{metrics[i][1]}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('<div class="result-banner result-feasible">✅ Feasible Feature Combination</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="gauges-container" style="text-align:center; padding:60px 20px; color:#94a3b8;">📊<br>Ready to Analyze<br><div style="font-size:0.95rem;">Select feature values above and click <b>START</b> to view results</div></div>', unsafe_allow_html=True)

st.markdown("<div class='footer-text'>© The Prototyped Decision Support System</div>", unsafe_allow_html=True)
