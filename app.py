import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="The Prototyped Decision Support System",
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

st.markdown('<div class="main-title">The Prototyped Decision Support System</div>', unsafe_allow_html=True)


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
    st.markdown("### Select a value for each feature")

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
            <div style="font-size:0.8rem; color:#6b7280; margin-bottom:2px;">Current feature combination </div>
            <div style="font-family:monospace; font-weight:600; color:#1f2937; font-size:0.85rem;">[{', '.join(feat_values)}]</div>
        </div>""",
        unsafe_allow_html=True
    )

    analyze = st.button("START")

# --------------------------
# 右侧：结果展示
# --------------------------
with right_col:
    if analyze:

        # ======================
        # Step 1: 可行性规则判断
        # ======================
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
        # Step 2: 如果不可行
        # ======================
        if infeasible:
            g1, g2, g3 = st.columns(3)

            def empty_gauge(title):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=0,
                    number={"font": {"size": 24}},
                    title={"text": title},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#9ca3af"}
                    }
                ))
                fig.update_layout(height=260)
                return fig

            with g1:
                st.plotly_chart(empty_gauge("均值"), use_container_width=True)

            with g2:
                st.plotly_chart(empty_gauge(">0 占比"), use_container_width=True)

            with g3:
                st.plotly_chart(empty_gauge("<0 占比"), use_container_width=True)

            # 红色提示（居中）
            st.markdown(
                """
                <div style="text-align:center; color:red; font-weight:700; font-size:1.2rem; margin-top:20px;">
                Infeasible Feature Combination
                </div>
                """,
                unsafe_allow_html=True
            )

        # ======================
        # Step 3: 可行情况
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

                    # ======================
                    # 统计
                    # ======================
                    mean_val = float(np.mean(all_results))
                    count = len(all_results)

                    pos_ratio = np.sum(all_results > 0) / count
                    neg_ratio = np.sum(all_results < 0) / count

                    # ======================
                    # 仪表盘函数（字体缩小）
                    # ======================
                    def draw_gauge(title, value, min_val, max_val, color, is_percent=False):
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            number={
                                "font": {"size": 24},  # ⭐缩小
                                "suffix": "%" if is_percent else ""
                            },
                            title={"text": title},
                            gauge={
                                "axis": {"range": [min_val, max_val]},
                                "bar": {"color": color},
                                "bgcolor": "white"
                            }
                        ))
                        fig.update_layout(height=260)
                        return fig

                    g1, g2, g3 = st.columns(3)

                    # ======================
                    # 均值仪表盘
                    # ======================
                    with g1:
                        st.markdown(
                            "<div style='text-align:center; font-size:0.9rem; color:#374151; margin-bottom:6px;'>"
                            "Predicted change in train delay per section if overridden"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    
                        st.plotly_chart(
                            draw_gauge(
                                "Mean",
                                mean_val,
                                min(0, np.min(all_results)),
                                np.max(all_results),
                                "#3b82f6"
                            ),
                            use_container_width=True
                        )
                    
                    # ======================
                    # >0 占比
                    # ======================
                    with g2:
                        st.markdown(
                            "<div style='text-align:center; font-size:0.9rem; color:#374151; margin-bottom:6px;'>"
                            "Predicted probability that override improves train punctuality"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    
                        st.plotly_chart(
                            draw_gauge(
                                ">0 Ratio",
                                pos_ratio * 100,
                                0, 100,
                                "#059669",
                                True
                            ),
                            use_container_width=True
                        )
                    
                    # ======================
                    # <0 占比
                    # ======================
                    with g3:
                        st.markdown(
                            "<div style='text-align:center; font-size:0.9rem; color:#374151; margin-bottom:6px;'>"
                            "Predicted probability that override harms train punctuality"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    
                        st.plotly_chart(
                            draw_gauge(
                                "<0 Ratio",
                                neg_ratio * 100,
                                0, 100,
                                "#dc2626",
                                True
                            ),
                            use_container_width=True
                        )

                    # 居中提示（在所有仪表盘下面）
                    st.markdown(
                        """
                        <div style="text-align:center; color:#059669; font-weight:700; font-size:1.1rem; margin-top:20px;">
                        ✅ Feasible Feature Combination
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9ca3af; font-size:0.9rem;'>The Prototyped Decision Support System</div>",
            unsafe_allow_html=True)
