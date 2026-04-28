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
# 样式（排版优化）
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== 标题区域 ===== */
.title-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 40px 20px 30px 20px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.25);
    position: relative;
    overflow: hidden;
}

.title-wrapper::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #3b82f6);
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.sub-title {
    font-size: 1.1rem;
    text-align: center;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ===== 区块样式 ===== */
.section-block {
    background: #ffffff;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.feature-label {
    text-align: center;
    font-weight: 700;
    font-size: 0.8rem;
    color: #1e293b;
    margin-bottom: 4px;
}

.combo-box {
    background: #f0f9ff;
    border-left: 5px solid #0ea5e9;
    padding: 14px;
    border-radius: 8px;
    margin-top: 14px;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.08);
}

/* ===== 仪表盘卡片 ===== */
.gauge-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px 10px 24px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.gauge-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}

.gauge-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f1f5f9;
}

.gauge-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 12px;
    font-size: 1.1rem;
}

.gauge-title-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
}

.gauge-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2px;
}

/* ===== 结果状态 ===== */
.result-banner {
    text-align: center;
    font-weight: 800;
    font-size: 1.3rem;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
    letter-spacing: 0.3px;
}

.result-feasible {
    background: #ecfdf5;
    color: #047857;
    border: 2px solid #6ee7b7;
}

.result-infeasible {
    background: #fef2f2;
    color: #b91c1c;
    border: 2px solid #fca5a5;
}

/* ===== 按钮 ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
}

/* ===== Footer ===== */
.footer-text {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 20px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 标题区域
# ======================
st.markdown("""
<div class="title-wrapper">
    <div class="main-title">The Prototyped Decision Support System</div>
    <div class="sub-title">Train Delay Override Analysis &amp; Punctuality Prediction Dashboard</div>
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
# 左右布局
# ======================
left_col, right_col = st.columns([0.45, 2.55])

# ======================
# 左侧：特征选择
# ======================
with left_col:

    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.markdown("### Select a value for each feature")

    selected_features = {}

    for row in range(3):
        cols = st.columns(5)
        for col_idx in range(5):
            feat_num = row * 5 + col_idx + 1
            if feat_num <= 15:
                with cols[col_idx]:
                    st.markdown(
                        f"<div class='feature-label'>F{feat_num}</div>",
                        unsafe_allow_html=True
                    )
                    selected_features[f"feature_{feat_num}"] = st.radio(
                        label=f"feature_{feat_num}",
                        options=[0, 1],
                        index=0,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"select_{feat_num}"
                    )

    st.markdown('</div>', unsafe_allow_html=True)

    # 当前组合
    feat_values = [str(selected_features[f"feature_{i}"]) for i in range(1, 16)]

    st.markdown(
        f"""<div class="combo-box">
            <div style="color:#475569; margin-bottom:4px; font-weight:600;">Current Feature Combination</div>
            <div style="font-family:'SF Mono', monospace; font-weight:700; color:#0f172a; font-size:0.95rem;">[{', '.join(feat_values)}]</div>
        </div>""",
        unsafe_allow_html=True
    )

    analyze = st.button("▶ START ANALYSIS", use_container_width=True)

# ======================
# 右侧：结果展示（仪表盘每行一个）
# ======================
with right_col:

    if analyze:

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

        # ======================
        # 不可行
        # ======================
        if infeasible:

            def empty_gauge_card(title, subtitle, icon, icon_bg, chart_key):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=0,
                    number={"font": {"size": 32, "color": "#94a3b8"}},
                    title={"text": ""},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#cbd5e1"},
                        "bar": {"color": "#cbd5e1"},
                        "bgcolor": "#f8fafc",
                        "bordercolor": "#e2e8f0"
                    }
                ))
                fig.update_layout(
                    height=280,
                    margin=dict(l=40, r=40, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.markdown(f"""
                <div class="gauge-card">
                    <div class="gauge-header">
                        <div class="gauge-icon" style="background:{icon_bg};">{icon}</div>
                        <div>
                            <div class="gauge-title-text">{title}</div>
                            <div class="gauge-subtitle">{subtitle}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key=chart_key)

            empty_gauge_card(
                "Predicted Change in Train Delay",
                "Mean delay variation per section if overridden",
                "📊", "#dbeafe",
                "empty_gauge_1"
            )
            empty_gauge_card(
                "Improvement Probability",
                "Predicted probability that override improves punctuality",
                "📈", "#d1fae5",
                "empty_gauge_2"
            )
            empty_gauge_card(
                "Harm Probability",
                "Predicted probability that override harms punctuality",
                "📉", "#fee2e2",
                "empty_gauge_3"
            )

            st.markdown(
                '<div class="result-banner result-infeasible">⚠️ Infeasible Feature Combination</div>',
                unsafe_allow_html=True
            )

        # ======================
        # 可行
        # ======================
        else:

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

                if len(all_results) == 0:
                    st.error("❌ 未找到模型结果")
                else:
                    all_results = np.array(all_results)

                    mean_val = float(np.mean(all_results))
                    count = len(all_results)
                    pos_ratio = np.sum(all_results > 0) / count
                    neg_ratio = np.sum(all_results < 0) / count

                    def draw_gauge_card(title, subtitle, value, min_val, max_val, color, icon, icon_bg, chart_key, is_percent=False):
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            number={"font": {"size": 36, "color": color, "weight": "bold"}, "suffix": "%" if is_percent else ""},
                            title={"text": ""},
                            gauge={
                                "axis": {"range": [min_val, max_val], "tickcolor": "#94a3b8"},
                                "bar": {"color": color, "thickness": 0.75},
                                "bgcolor": "#f1f5f9",
                                "bordercolor": "#e2e8f0",
                                "threshold": {
                                    "line": {"color": "#1e293b", "width": 3},
                                    "thickness": 0.8,
                                    "value": value
                                }
                            }
                        ))
                        fig.update_layout(
                            height=300,
                            margin=dict(l=40, r=40, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.markdown(f"""
                        <div class="gauge-card">
                            <div class="gauge-header">
                                <div class="gauge-icon" style="background:{icon_bg};">{icon}</div>
                                <div>
                                    <div class="gauge-title-text">{title}</div>
                                    <div class="gauge-subtitle">{subtitle}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)

                    # Gauge 1: Mean
                    draw_gauge_card(
                        "Predicted Change in Train Delay",
                        "Mean delay variation per section if overridden",
                        mean_val,
                        min(0, np.min(all_results)),
                        np.max(all_results),
                        "#2563eb",
                        "📊", "#dbeafe",
                        "gauge_mean",
                        False
                    )

                    # Gauge 2: Positive Ratio
                    draw_gauge_card(
                        "Improvement Probability",
                        "Predicted probability that override improves train punctuality",
                        pos_ratio * 100,
                        0, 100,
                        "#059669",
                        "📈", "#d1fae5",
                        "gauge_pos",
                        True
                    )

                    # Gauge 3: Negative Ratio
                    draw_gauge_card(
                        "Harm Probability",
                        "Predicted probability that override harms train punctuality",
                        neg_ratio * 100,
                        0, 100,
                        "#dc2626",
                        "📉", "#fee2e2",
                        "gauge_neg",
                        True
                    )

                    st.markdown(
                        '<div class="result-banner result-feasible">✅ Feasible Feature Combination</div>',
                        unsafe_allow_html=True
                    )

# Footer
st.markdown("<div class='footer-text'>The Prototyped Decision Support System</div>", unsafe_allow_html=True)
