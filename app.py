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
    layout="wide"
)

# ======================
# 样式（增强版）
# ======================
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 16px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
    color: #374151;
}

.feature-label {
    text-align: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: #374151;
}

.combo-box {
    background:#eff6ff;
    border-left:4px solid #3b82f6;
    padding:10px;
    border-radius:6px;
    margin-top:10px;
    font-size:0.9rem;
}

.metric-title {
    text-align:center;
    font-size:0.9rem;
    color:#374151;
    margin-bottom:6px;
}

.result-text {
    text-align:center;
    font-weight:700;
    font-size:1.2rem;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 标题
# ======================
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
# Feature Selection（整块上移）
# ======================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Select a value for each feature</div>', unsafe_allow_html=True)

selected_features = {}

for row in range(3):
    cols = st.columns(5)
    for col_idx in range(5):
        feat_num = row * 5 + col_idx + 1
        if feat_num <= 15:
            with cols[col_idx]:
                st.markdown(f"<div class='feature-label'>F{feat_num}</div>", unsafe_allow_html=True)
                selected_features[f"feature_{feat_num}"] = st.radio(
                    f"feature_{feat_num}",
                    [0, 1],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"select_{feat_num}"
                )

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 当前组合 + 按钮（单独一行）
# ======================
feat_values = [str(selected_features[f"feature_{i}"]) for i in range(1, 16)]

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        f"""<div class="combo-box">
        <div style="color:#6b7280;">Current feature combination</div>
        <div style="font-family:monospace; font-weight:600;">[{', '.join(feat_values)}]</div>
        </div>""",
        unsafe_allow_html=True
    )

with col2:
    analyze = st.button("START")

# ======================
# 结果区（独立大卡片）
# ======================
if analyze:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    g1, g2, g3 = st.columns(3)

    # 不可行
    if infeasible:

        def empty_gauge(title):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                number={"font": {"size": 24}},
                title={"text": ""},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#9ca3af"}}
            ))
            fig.update_layout(height=260)
            return fig

        with g1:
            st.markdown('<div class="metric-title">Mean</div>', unsafe_allow_html=True)
            st.plotly_chart(empty_gauge(""), use_container_width=True)

        with g2:
            st.markdown('<div class="metric-title">&gt;0 Ratio</div>', unsafe_allow_html=True)
            st.plotly_chart(empty_gauge(""), use_container_width=True)

        with g3:
            st.markdown('<div class="metric-title">&lt;0 Ratio</div>', unsafe_allow_html=True)
            st.plotly_chart(empty_gauge(""), use_container_width=True)

        st.markdown('<div class="result-text" style="color:red;">Infeasible Feature Combination</div>', unsafe_allow_html=True)

    # 可行
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

                def draw_gauge(value, min_val, max_val, color, is_percent=False):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        number={"font": {"size": 24}, "suffix": "%" if is_percent else ""},
                        title={"text": ""},
                        gauge={"axis": {"range": [min_val, max_val]}, "bar": {"color": color}}
                    ))
                    fig.update_layout(height=260)
                    return fig

                with g1:
                    st.markdown('<div class="metric-title">Predicted change in train delay per section if overridden</div>', unsafe_allow_html=True)
                    st.plotly_chart(draw_gauge(mean_val, min(0, np.min(all_results)), np.max(all_results), "#3b82f6"), use_container_width=True)

                with g2:
                    st.markdown('<div class="metric-title">Predicted probability that override improves train punctuality</div>', unsafe_allow_html=True)
                    st.plotly_chart(draw_gauge(pos_ratio * 100, 0, 100, "#059669", True), use_container_width=True)

                with g3:
                    st.markdown('<div class="metric-title">Predicted probability that override harms train punctuality</div>', unsafe_allow_html=True)
                    st.plotly_chart(draw_gauge(neg_ratio * 100, 0, 100, "#dc2626", True), use_container_width=True)

                st.markdown('<div class="result-text" style="color:#059669;">Feasible Feature Combination</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9ca3af; font-size:0.9rem;'>The Prototyped Decision Support System</div>", unsafe_allow_html=True)
