import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="结果可视化分析",
    page_icon="🎯",
    layout="wide"
)

# 自定义样式
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1f2937; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #6b7280; text-align: center; margin-bottom: 2rem; }
    .stButton>button { width: 100%; height: 2.8rem; background: #3b82f6; color: white; font-weight: 600; border-radius: 8px; border: none; }
    .stButton>button:hover { background: #2563eb; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 模型结果可视化分析</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">选择15个特征值，查看1600个二叉树模型的预测结果分布</div>', unsafe_allow_html=True)


# --------------------------
# 后台加载数据
# --------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "feature_combinations.csv"))
    df_results = pd.read_csv(os.path.join(base_dir, "beta_merged_processed_0418.csv"))
    return df_features, df_results


df_features, df_results = load_data()

# --------------------------
# 左侧：特征选择
# --------------------------
left_col, right_col = st.columns([0.45, 2.55])

with left_col:
    st.markdown("### 🔧 特征选择")
    st.caption("为每个特征选择 0 或 1")

    selected_features = {}

    for row in range(3):
        cols = st.columns(5)
        for col_idx in range(5):
            feat_num = row * 5 + col_idx + 1
            if feat_num <= 15:
                with cols[col_idx]:
                    st.markdown(
                        f"<p style='text-align:center; font-weight:600; color:#374151; margin-bottom:2px; font-size:0.75rem;'>F{feat_num}</p>",
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

    feat_values = [str(selected_features[f"feature_{i}"]) for i in range(1, 16)]
    st.markdown(
        f"""<div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:10px; border-radius:6px; margin-top:12px;">
            <div style="font-size:0.8rem; color:#6b7280; margin-bottom:2px;">当前特征组合</div>
            <div style="font-family:monospace; font-weight:600; color:#1f2937; font-size:0.85rem;">[{', '.join(feat_values)}]</div>
        </div>""",
        unsafe_allow_html=True
    )

    analyze = st.button("🚀 分析模型结果")

# --------------------------
# 右侧：结果展示
# --------------------------
with right_col:
    if analyze:
        match_mask = pd.Series([True] * len(df_features))
        for feat_name, feat_val in selected_features.items():
            match_mask = match_mask & (df_features[feat_name] == feat_val)

        matched_ids = df_features.loc[match_mask, "id"].values

        if len(matched_ids) == 0:
            st.error("❌ 未找到完全匹配的特征组合")
            feature_cols = [f"feature_{i}" for i in range(1, 16)]
            sel_vec = np.array([selected_features[c] for c in feature_cols])
            hamming_dist = np.sum(np.abs(df_features[feature_cols].values - sel_vec), axis=1)
            df_temp = df_features.copy()
            df_temp["差异数"] = hamming_dist
            nearest = df_temp.nsmallest(5, "差异数")[["id"] + feature_cols + ["差异数"]]
            st.dataframe(nearest.reset_index(drop=True), use_container_width=True)

        else:
            all_results = []
            for mid in matched_ids:
                if mid in df_results["id"].values:
                    row_vals = df_results.loc[df_results["id"] == mid].drop(columns=["id"]).values.flatten()
                    all_results.extend(row_vals)

            if len(all_results) == 0:
                st.error("❌ 特征匹配成功，但未找到对应的模型结果数据")
            else:
                all_results = np.array(all_results)

                # ======================
                # 核心统计
                # ======================
                mean_val = float(np.mean(all_results))
                positive_count = int(np.sum(all_results > 0))
                count = len(all_results)

                pos_ratio = positive_count / count if count > 0 else 0

                # ======================
                # 第一行：仪表盘（均值 + >0占比）
                # ======================
                st.markdown("### 🎯 模型分析仪表盘")

                g1, g2 = st.columns(2)

                # 1. 均值仪表盘
                with g1:
                    gauge_min = min(0, min(all_results) * 1.1) if min(all_results) < 0 else 0
                    gauge_max = max(all_results) * 1.1 if max(all_results) != 0 else 1

                    fig_mean = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=mean_val,
                        number={"font": {"size": 36}, "valueformat": ".2f"},
                        title={"text": "预测结果均值"},
                        gauge={
                            "axis": {"range": [gauge_min, gauge_max]},
                            "bar": {"color": "#3b82f6"},
                            "bgcolor": "white"
                        }
                    ))
                    fig_mean.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
                    st.plotly_chart(fig_mean, use_container_width=True)

                # 2. 正值计数仪表盘
                with g2:
                    fig_count = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pos_ratio * 100,
                        number={"font": {"size": 36}, "suffix": "%"},
                        title={"text": "大于0的预测值占比"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#059669"},  # 绿色
                            "bgcolor": "white",
                            "steps": [{"range": [0, 100], "color": "#f0fdf4"}]
                        }
                    ))
                    fig_count.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
                    st.plotly_chart(fig_count, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#9ca3af; font-size:0.9rem;'>模型结果可视化分析平台 | Streamlit + Plotly</div>",
            unsafe_allow_html=True)
