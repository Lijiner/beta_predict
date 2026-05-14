import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="The Prototyped Decision Support System", layout="wide")

# ======================
# 样式
# ======================
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: bold; text-align: center; }
.sub-title { font-size: 1rem; text-align: center; margin-bottom: 1rem; }
.feature-box {
    padding: 6px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    text-align: center;
    margin-bottom: 10px;
}
.feature-label {
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.2;
    margin-bottom: 6px;
}
.center-text { text-align:center; margin-top:10px; font-size:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">The Prototyped Decision Support System</div>', unsafe_allow_html=True)

# ======================
# 数据加载（改造：加载预计算统计结果）
# ======================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "feature_combinations.csv"))
    # 加载由 process_beta_data.py 生成的预计算统计结果
    df_stats = pd.read_csv(os.path.join(base_dir, "beta2000_statistics.csv"))
    return df_features, df_stats

df_features, df_stats = load_data()

# 确定统计文件的 id 列（第一列）
stats_id_col = df_stats.columns[0]

# ======================
# Fragment 兼容
# ======================
try:
    fragment = st.fragment
except AttributeError:
    try:
        fragment = st.experimental_fragment
    except AttributeError:
        def fragment(func):
            return func

# ======================
# Feature 定义（名称 + 描述）
# ======================
FEATURE_DEFS = [
    ("F1",  "number_incident = 0"),
    ("F2",  "length_incident = 0"),
    ("F3",  "length_incident <= 3"),
    ("F4",  "density <= 0.15"),
    ("F5",  "density <= 0.28"),
    ("F6",  "density <= 0.45"),
    ("F7",  "conflict = 0"),
    ("F8",  "conflict <= 0.25"),
    ("F9",  "redlights = 0"),
    ("F10", "redlights <= 0.25"),
    ("F11", "zonehour_typical_workload = 1"),
    ("F12", "zonehour_typical_workload <= 2"),
    ("F13", "zonehour_typical_workload <= 3"),
    ("F14", "zonehour_typical_workload <= 4"),
    ("F15", "peak = 1"),
]

# ======================
# 特征选择（3 行 × 5 列 布局）
# ======================
@fragment
def render_feature_selector():
    st.markdown("### Select a value for each feature")

    selected_features = {}

    # 3 行，每行 5 列
    for row_idx in range(3):
        cols = st.columns(5)
        for col_idx in range(5):
            feat_idx = row_idx * 5 + col_idx      # 0-based
            feat_num = feat_idx + 1                 # 1-based (F1~F15)
            feat_key = f"feature_{feat_num}"
            feat_label, feat_desc = FEATURE_DEFS[feat_idx]

            with cols[col_idx]:
                st.markdown(
                    f"<div class='feature-box'>"
                    f"<div class='feature-label'>{feat_label}: {feat_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                selected_features[feat_key] = st.radio(
                    label=f"feature_{feat_num}",
                    options=[False, True],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"f{feat_num}"
                )

    feat_values = [str(selected_features[f"feature_{i}"]) for i in range(1, 16)]
    st.markdown(
        f"""<div style="background:#f3f4f6; padding:8px; border-radius:6px; margin-top:10px;">
        Current feature combination: <b>[{', '.join(feat_values)}]</b>
        </div>""",
        unsafe_allow_html=True
    )

    return selected_features

selected_features = render_feature_selector()

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("START")

# ======================
# 规则校验（自动将 bool 转为 0/1）
# ======================
def is_feasible(f):
    # 统一转为 0/1，兼容 CSV 数据格式
    fv = {k: (1 if v else 0) for k, v in f.items()}

    if fv["feature_2"] == 1 and fv["feature_3"] != 1: return False
    if fv["feature_4"] == 1 and (fv["feature_5"] != 1 or fv["feature_6"] != 1): return False
    if fv["feature_5"] == 1 and fv["feature_6"] != 1: return False
    if fv["feature_7"] == 1 and fv["feature_8"] != 1: return False
    if fv["feature_9"] == 1 and fv["feature_10"] != 1: return False
    if fv["feature_11"] == 1 and (fv["feature_12"] != 1 or fv["feature_13"] != 1 or fv["feature_14"] != 1): return False
    if fv["feature_12"] == 1 and (fv["feature_13"] != 1 or fv["feature_14"] != 1): return False
    if fv["feature_13"] == 1 and fv["feature_14"] != 1: return False
    return True

# ======================
# 仪表盘
# ======================
if analyze:

    feasible = is_feasible(selected_features)

    if feasible:
        # 在 feature_combinations 中找到完全匹配的行
        match_mask = pd.Series([True] * len(df_features))
        for k, v in selected_features.items():
            match_mask &= (df_features[k] == (1 if v else 0))

        matched_ids = df_features.loc[match_mask, "id"]

        if len(matched_ids) == 0:
            st.error("No matching feature combination found in the dataset.")
            mean_val = 0.0
            pos_ratio = 0.0
            neg_ratio = 0.0
        else:
            # 通过 id 从预计算统计文件中读取指标
            # 统一为字符串进行比较，避免类型不匹配
            target_id = str(matched_ids.iloc[0])
            df_stats[stats_id_col] = df_stats[stats_id_col].astype(str)
            stat_row = df_stats[df_stats[stats_id_col] == target_id]

            if len(stat_row) == 0:
                st.error("Pre-computed statistics not found for this combination.")
                mean_val = 0.0
                pos_ratio = 0.0
                neg_ratio = 0.0
            else:
                row = stat_row.iloc[0]
                mean_val = float(row["mean"])
                pos_ratio = float(row["greater_0"])
                neg_ratio = float(row["less_0"])
    else:
        mean_val = 0.0
        pos_ratio = 0.0
        neg_ratio = 0.0

    gauge_min, gauge_max = -100, 100

    g1, g2, g3 = st.columns(3)

    # ======================
    # G1
    # ======================
    with g1:
        st.markdown("<div style='text-align:center;font-weight:700;font-size:20px;'>Predicted probability that override harms train punctuality</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(pos_ratio) * 100,
            number={"suffix": "%", "font": {"size": 25}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickfont": {"size": 16}, "showticklabels": True},
                "bar": {"color": "#dc2626", "thickness": 0.25}
            }
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # G2
    # ======================
    with g2:
        st.markdown("<div style='text-align:center;font-weight:700;font-size:20px;'>Predicted change in train delay if overriden</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(mean_val),
            number={"suffix":" seconds per train section","valueformat": ".2f", "font": {"size": 25}},
            gauge={
                "shape": "angular",
                "axis": {"range": [gauge_min, gauge_max], "tickfont": {"size": 16}, "showticklabels": True},
                "bar": {"color": "#2563eb", "thickness": 0.25}
            }
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # G3
    # ======================
    with g3:
        st.markdown("<div style='text-align:center;font-weight:700;font-size:20px;'>Predicted probability that override improves train punctuality</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(neg_ratio) * 100,
            number={"suffix": "%", "font": {"size": 25}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickfont": {"size": 16}, "showticklabels": True},
                "bar": {"color": "#059669", "thickness": 0.25}
            }
        ))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 可行性提示（文案已更新）
    # ======================
    if feasible:
        st.markdown("<div style='text-align:center;font-size:28px;'><b>Feasible scenario （Decision support provided）</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;color:red;font-size:28px;'><b>Infeasible scenario （Decision support not provided）</b></div>", unsafe_allow_html=True)

st.markdown("---")
