import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="The Prototyped Decision Support System",
    layout="wide"
)

# =========================
# 样式
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
    }

    .sub-title {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    .stButton>button {
        width: 100%;
        height: 2.8rem;
        background: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
    }

    .stButton>button:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">The Prototyped Decision Support System</div>', unsafe_allow_html=True)

# =========================
# 数据加载
# =========================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "feature_combinations.csv"))
    df_results = pd.read_csv(os.path.join(base_dir, "beta_merged_processed_0418.csv"))
    return df_features, df_results


df_features, df_results = load_data()

# =========================
# 左侧输入
# =========================
left_col, right_col = st.columns([0.45, 2.55])

with left_col:
    st.markdown("### Select a value for each feature")

    selected_features = {}

    for row in range(3):
        cols = st.columns(5)
        for col_idx in range(5):
            feat_num = row * 5 + col_idx + 1
            if feat_num <= 15:
                with cols[col_idx]:
                    st.markdown(
                        f"<p style='text-align:center; font-weight:600; font-size:0.75rem;'>F{feat_num}</p>",
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

    analyze = st.button("START")

# =========================
# 半圆仪表盘（核心）
# =========================
def draw_semi_gauge(title, value, min_val, max_val, color, is_percent=False):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "font": {"size": 48},
            "suffix": "%" if is_percent else ""
        },
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 20}
        },
        gauge={
            "shape": "angular",

            "axis": {
                "range": [min_val, max_val]
            },

            "bar": {"color": color},

            "steps": [
                {"range": [min_val, (min_val + max_val) / 2], "color": "#e5e7eb"},
                {"range": [(min_val + max_val) / 2, max_val], "color": "#f3f4f6"}
            ]
        }
    ))

    fig.update_layout(
        height=380,
        margin=dict(t=30, b=0, l=10, r=10)
    )

    return fig

# =========================
# 右侧结果
# =========================
with right_col:
    if analyze:

        f = selected_features

        # =========================
        # 可行性判断
        # =========================
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

        # =========================
        # 不可行
        # =========================
        if infeasible:
            g1, g2, g3 = st.columns(3)

            def empty():
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=0,
                    number={"font": {"size": 40}},
                    gauge={"axis": {"range": [0, 100]}}
                ))
                fig.update_layout(height=320)
                return fig

            with g1:
                st.plotly_chart(empty(), use_container_width=True)
            with g2:
                st.plotly_chart(empty(), use_container_width=True)
            with g3:
                st.plotly_chart(empty(), use_container_width=True)

            st.markdown(
                "<div style='text-align:center;color:red;font-size:1.2rem;font-weight:700;'>"
                "Infeasible Feature Combination"
                "</div>",
                unsafe_allow_html=True
            )

        # =========================
        # 可行情况
        # =========================
        else:

            match_mask = pd.Series([True] * len(df_features))
            for feat_name, feat_val in selected_features.items():
                match_mask &= (df_features[feat_name] == feat_val)

            matched_ids = df_features.loc[match_mask, "id"].values

            all_results = []

            for mid in matched_ids:
                if mid in df_results["id"].values:
                    row_vals = df_results.loc[df_results["id"] == mid].drop(columns=["id"]).values.flatten()
                    all_results.extend(row_vals)

            all_results = np.array(all_results)

            mean_val = float(np.mean(all_results))
            pos_ratio = np.mean(all_results > 0)
            neg_ratio = np.mean(all_results < 0)

            # =========================
            # 标题
            # =========================
            st.markdown("""
            <div style='text-align:center;font-size:1.3rem;font-weight:700;color:#111827;margin-bottom:10px;'>
                Decision Cockpit Dashboard
            </div>
            """, unsafe_allow_html=True)

            # =========================
            # 第一排：核心指标（半圆）
            # =========================
            st.markdown("""
            <div style='text-align:center;font-size:1.05rem;font-weight:600;color:#374151;'>
                System-Level Impact
            </div>
            """, unsafe_allow_html=True)

            col_center = st.columns([1, 2, 1])[1]

            with col_center:
                st.plotly_chart(
                    draw_semi_gauge(
                        "Mean Delay Impact",
                        mean_val,
                        min(0, np.min(all_results)),
                        np.max(all_results),
                        "#3b82f6"
                    ),
                    use_container_width=True
                )

            # =========================
            # 第二排：风险拆解
            # =========================
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("""
                <div style='text-align:center;font-size:1rem;font-weight:600;'>
                    Improvement Probability
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(
                    draw_semi_gauge(
                        "Improve %",
                        pos_ratio * 100,
                        0, 100,
                        "#059669",
                        True
                    ),
                    use_container_width=True
                )

            with g2:
                st.markdown("""
                <div style='text-align:center;font-size:1rem;font-weight:600;'>
                    Risk Probability
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(
                    draw_semi_gauge(
                        "Harm %",
                        neg_ratio * 100,
                        0, 100,
                        "#dc2626",
                        True
                    ),
                    use_container_width=True
                )

            # =========================
            # 总结条
            # =========================
            st.markdown(f"""
            <div style='text-align:center;margin-top:15px;font-size:1.05rem;color:#374151;'>
                Mean: <b>{mean_val:.3f}</b> |
                Improve: <b>{pos_ratio*100:.1f}%</b> |
                Risk: <b>{neg_ratio*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='text-align:center;color:#059669;font-weight:700;font-size:1.1rem;margin-top:10px;'>
                Feasible Feature Combination
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;'>The Prototyped Decision Support System</div>",
    unsafe_allow_html=True
)
