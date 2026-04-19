import sys
if hasattr(sys, '_MEIPASS'):
    # PyInstaller 打包后的路径处理
    import os
    os.chdir(sys._MEIPASS)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
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
    .stat-card { background: #f9fafb; border-radius: 12px; padding: 16px; text-align: center; border-top: 3px solid #3b82f6; }
    .stat-label { font-size: 0.85rem; color: #6b7280; }
    .stat-value { font-size: 1.8rem; font-weight: bold; }
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
                mean_val = float(np.mean(all_results))
                median_val = float(np.median(all_results))
                std_val = float(np.std(all_results))
                min_val = float(np.min(all_results))
                max_val = float(np.max(all_results))
                count = len(all_results) -1
                positive_count = int(np.sum(all_results > 0))

                # 统计卡片
                st.markdown("### 📊 统计摘要")
                s1, s2, s3, s4, s5 = st.columns(5)
                stats = [
                    ("均值", f"{mean_val:.2f}", "#3b82f6"),
                    ("中位数", f"{median_val:.2f}", "#10b981"),
                    ("标准差", f"{std_val:.2f}", "#f59e0b"),
                    ("预测值数量", f"{count}", "#8b5cf6"),
                    ("预测值大于0的数量",f"{positive_count}","#059669")
                ]
                for col, (lab, val, color) in zip([s1, s2, s3, s4,s5], stats):
                    col.markdown(
                        f"""<div class="stat-card" style="border-top-color: {color};">
                            <div class="stat-label">{lab}</div>
                            <div class="stat-value" style="color: {color};">{val}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # ---------- 圆形仪表盘 ----------
                st.markdown("### 🎯 模型结果均值仪表盘")

                gauge_min = min(0, min_val * 1.1) if min_val < 0 else 0
                gauge_max = max_val * 1.1
                zero_ratio = (0 - gauge_min) / (gauge_max - gauge_min) if gauge_max != gauge_min else 0.5

                # 0位置线（占量程1.5%的琥珀色窄条）
                zero_band = (gauge_max - gauge_min) * 0.015
                zero_left = max(gauge_min, 0 - zero_band / 2)
                zero_right = min(gauge_max, 0 + zero_band / 2)

                steps = []
                if gauge_min < zero_left:
                    steps.append({"range": [gauge_min, zero_left], "color": "rgba(239, 68, 68, 0.12)"})
                steps.append({"range": [zero_left, zero_right], "color": "#f59e0b"})
                if zero_right < gauge_max:
                    steps.append({"range": [zero_right, gauge_max], "color": "rgba(59, 130, 246, 0.12)"})

                fig_gauge = go.Figure()

                fig_gauge.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=mean_val,
                        number={
                            "font": {"size": 48, "color": "#1e3a8a", "family": "Arial Black"},
                            "valueformat": ".2f"
                        },
                        title={
                            "text": f"<b>均值</b><br><span style='font-size:13px;color:#6b7280'>基于 {count} 个模型结果</span>",
                            "font": {"size": 18}
                        },
                        gauge={
                            "axis": {"range": [gauge_min, gauge_max], "tickwidth": 2, "tickcolor": "#374151",
                                     "tickformat": ".0f"},
                            "bar": {"color": "#3b82f6", "thickness": 0.55},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "#d1d5db",
                            "steps": steps,
                            "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.75,
                                          "value": mean_val}
                        },
                        domain={"x": [0, 1], "y": [0, 1]}
                    )
                )

                # 标注放在0线正下方，箭头向上指向0线
                # if 0 <= zero_ratio <= 1:
                #     theta = np.pi * (1 - zero_ratio)
                #     arc_y = 0.15 + 0.35 * np.sin(theta)
                #     ann_y = arc_y - 0.14
                #     ann_y = max(0.08, ann_y)
                #
                #     fig_gauge.add_annotation(
                #         x=zero_ratio,
                #         y=ann_y,
                #         text=f"<b>0</b> ({positive_count}个正值)",
                #         showarrow=True,
                #         arrowhead=2,
                #         arrowsize=1,
                #         arrowwidth=2,
                #         arrowcolor="#f59e0b",
                #         ax=0,
                #         ay=15,  # 箭头向上指
                #         font=dict(size=12, color="#92400e"),
                #         bgcolor="rgba(254, 243, 199, 0.95)",
                #         bordercolor="#f59e0b",
                #         borderwidth=2,
                #         borderpad=4,
                #         align="center"
                #     )
                #
                # fig_gauge.update_layout(
                #     height=420,
                #     margin=dict(l=50, r=50, t=90, b=50),
                #     paper_bgcolor="rgba(0,0,0,0)",
                #     plot_bgcolor="rgba(0,0,0,0)"
                # )

                st.plotly_chart(fig_gauge, use_container_width=True)

                # ---------- 直方图（正负值bin完全分离） ----------
                st.markdown("### 📈 模型结果分布直方图")

                positive_results = all_results[all_results > 0]
                negative_results = all_results[all_results <= 0]

                hist_min = min(all_results) * 1.02
                hist_max = max(all_results) * 1.02
                n_bins = 40

                fig_hist = go.Figure()

                # 负值独立分bin：[hist_min, 0]
                if len(negative_results) > 0:
                    neg_counts, neg_edges = np.histogram(negative_results, bins=n_bins // 2, range=(hist_min, 0))
                    neg_centers = (neg_edges[:-1] + neg_edges[1:]) / 2
                    neg_width = neg_edges[1] - neg_edges[0]
                    mask_neg = neg_counts > 0
                    if np.any(mask_neg):
                        fig_hist.add_trace(go.Bar(
                            x=neg_centers[mask_neg],
                            y=neg_counts[mask_neg],
                            width=neg_width,
                            marker_color="#ef4444",
                            opacity=0.75,
                            name="≤0",
                            hovertemplate="区间: %{x:.2f}<br>频数: %{y}<extra></extra>",
                            marker_line_width=0
                        ))

                # 正值独立分bin：[0, hist_max]
                if len(positive_results) > 0:
                    pos_counts, pos_edges = np.histogram(positive_results, bins=n_bins // 2, range=(0, hist_max))
                    pos_centers = (pos_edges[:-1] + pos_edges[1:]) / 2
                    pos_width = pos_edges[1] - pos_edges[0]
                    mask_pos = pos_counts > 0
                    if np.any(mask_pos):
                        fig_hist.add_trace(go.Bar(
                            x=pos_centers[mask_pos],
                            y=pos_counts[mask_pos],
                            width=pos_width,
                            marker_color="#3b82f6",
                            opacity=0.75,
                            name=f">0 ({positive_count}个)",
                            hovertemplate="区间: %{x:.2f}<br>频数: %{y}<extra></extra>",
                            marker_line_width=0
                        ))

                # 0线
                fig_hist.add_vline(
                    x=0, line_width=3, line_dash="solid", line_color="#10b981",
                    annotation_text=f"0线 ({positive_count}个正值)",
                    annotation_position="top", annotation_font_color="#059669",
                    annotation_font_size=12, annotation_font_family="Arial Black"
                )

                # 均值线
                fig_hist.add_vline(
                    x=mean_val, line_width=2.5, line_dash="dash", line_color="#ef4444",
                    annotation_text=f"均值: {mean_val:.2f}", annotation_position="top",
                    annotation_font_color="#ef4444", annotation_font_size=11
                )

                # 中位数线
                fig_hist.add_vline(
                    x=median_val, line_width=2, line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"中位数: {median_val:.2f}", annotation_position="bottom",
                    annotation_font_color="#f59e0b", annotation_font_size=11
                )

                fig_hist.update_layout(
                    height=400,
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="模型预测值",
                    yaxis_title="频数",
                    barmode="group",
                    bargap=0.02,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#9ca3af; font-size:0.9rem;'>模型结果可视化分析平台 | Streamlit + Plotly</div>",
            unsafe_allow_html=True)
