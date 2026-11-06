
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="The Prototyped Decision Support System",
    layout="wide"
)

GAUGE_HEIGHT = 420

# ======================
# 样式
# ======================
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: bold;
    text-align: center;
}

.sub-title {
    font-size: 1rem;
    text-align: center;
    margin-bottom: 1rem;
}

.row-title {
    font-size: 16px;
    font-weight: 700;
    color: #374151;
    margin-top: 16px;
    margin-bottom: 8px;
    padding-left: 4px;
    border-left: 4px solid #2563eb;
}

.gauge-title {
    height: 108px;
    min-height: 108px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 8px;
    text-align: center;
    font-size: clamp(16px, 1.25vw, 20px);
    font-weight: 700;
    line-height: 1.25;
}

@media (max-width: 768px) {
    .gauge-title {
        height: 168px;
        min-height: 168px;
        font-size: 15px;
        line-height: 1.2;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">The Prototyped Decision Support System</div>',
    unsafe_allow_html=True
)


# ======================
# 数据加载
# ======================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    df_features = pd.read_csv(
        os.path.join(base_dir, "feature_combinations.csv")
    )

    df_stats = pd.read_csv(
        os.path.join(base_dir, "beta2000_statistics.csv")
    )

    df_webapp2 = pd.read_csv(
        os.path.join(base_dir, "webapp2.csv")
    )

    return df_features, df_stats, df_webapp2


df_features, df_stats, df_webapp2 = load_data()

stats_id_col = df_stats.columns[0]

FEATURE_COLS = [f"feature_{i}" for i in range(1, 16)]

MISSING_STATS_COLS = [
    col for col in ("mean", "greater_0", "less_0")
    if col not in df_stats.columns
]

MISSING_WEBAPP2_COLS = [
    col for col in FEATURE_COLS + ["tree_used_ratio"]
    if col not in df_webapp2.columns
]

if MISSING_STATS_COLS:
    st.error(
        f"beta2000_statistics.csv is missing columns: "
        f"{', '.join(MISSING_STATS_COLS)}"
    )
    st.stop()

if MISSING_WEBAPP2_COLS:
    st.error(
        f"webapp2.csv is missing columns: "
        f"{', '.join(MISSING_WEBAPP2_COLS)}"
    )
    st.stop()


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
# 类别定义
# 用户可见选项 → 后台特征映射
# ======================

# number_incident
NUMBER_INCIDENT_CATS = {
    "number_incident = 0": {
        "feature_1": True
    },
    "number_incident ≠ 0": {
        "feature_1": False
    },
}

# length_incident
LENGTH_INCIDENT_CATS = {
    "length_incident = 0": {
        "feature_2": True,
        "feature_3": True
    },
    "0 < length_incident ≤ 3": {
        "feature_2": False,
        "feature_3": True
    },
    "length_incident > 3": {
        "feature_2": False,
        "feature_3": False
    },
}

# density
DENSITY_CATS = {
    "density ≤ 0.15": {
        "feature_4": True,
        "feature_5": True,
        "feature_6": True
    },
    "0.15 < density ≤ 0.28": {
        "feature_4": False,
        "feature_5": True,
        "feature_6": True
    },
    "0.28 < density ≤ 0.45": {
        "feature_4": False,
        "feature_5": False,
        "feature_6": True
    },
    "density > 0.45": {
        "feature_4": False,
        "feature_5": False,
        "feature_6": False
    },
}

# conflict
CONFLICT_CATS = {
    "conflict = 0": {
        "feature_7": True,
        "feature_8": True
    },
    "0 < conflict ≤ 0.25": {
        "feature_7": False,
        "feature_8": True
    },
    "conflict > 0.25": {
        "feature_7": False,
        "feature_8": False
    },
}

# redlights
REDLIGHTS_CATS = {
    "redlights = 0": {
        "feature_9": True,
        "feature_10": True
    },
    "0 < redlights ≤ 0.25": {
        "feature_9": False,
        "feature_10": True
    },
    "redlights > 0.25": {
        "feature_9": False,
        "feature_10": False
    },
}

# zonehour_typical_workload
WORKLOAD_CATS = {
    "zonehour_typical_workload ≤ 1": {
        "feature_11": True,
        "feature_12": True,
        "feature_13": True,
        "feature_14": True
    },
    "1 < zonehour_typical_workload ≤ 2": {
        "feature_11": False,
        "feature_12": True,
        "feature_13": True,
        "feature_14": True
    },
    "2 < zonehour_typical_workload ≤ 3": {
        "feature_11": False,
        "feature_12": False,
        "feature_13": True,
        "feature_14": True
    },
    "3 < zonehour_typical_workload ≤ 4": {
        "feature_11": False,
        "feature_12": False,
        "feature_13": False,
        "feature_14": True
    },
    "zonehour_typical_workload > 4": {
        "feature_11": False,
        "feature_12": False,
        "feature_13": False,
        "feature_14": False
    },
}

# peak
PEAK_CATS = {
    "peak = 1": {
        "feature_15": True
    },
    "peak ≠ 1": {
        "feature_15": False
    },
}


CATEGORY_GROUPS = [
    (
        "Select incident count category",
        NUMBER_INCIDENT_CATS,
        "cat_number_incident"
    ),
    (
        "Select incident length category",
        LENGTH_INCIDENT_CATS,
        "cat_length_incident"
    ),
    (
        "Select traffic density category",
        DENSITY_CATS,
        "cat_density"
    ),
    (
        "Select conflict level category",
        CONFLICT_CATS,
        "cat_conflict"
    ),
    (
        "Select red light category",
        REDLIGHTS_CATS,
        "cat_redlights"
    ),
    (
        "Select workload category",
        WORKLOAD_CATS,
        "cat_workload"
    ),
    (
        "Select peak hour category",
        PEAK_CATS,
        "cat_peak"
    ),
]


# ======================
# 类别选择器
# ======================
@fragment
def render_category_selector():

    selected_categories = {}

    for row_title, cat_dict, radio_key in CATEGORY_GROUPS:

        st.markdown(
            f"<div class='row-title'>{row_title}</div>",
            unsafe_allow_html=True
        )

        options = list(cat_dict.keys())

        selected_label = st.radio(
            label=radio_key,
            options=options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key=radio_key
        )

        selected_categories[radio_key] = selected_label

    return selected_categories


selected_categories = render_category_selector()

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button("START")


# ======================
# 类别转换为特征字典
# ======================
def categories_to_features(categories):

    features = {}

    all_cat_dicts = [
        NUMBER_INCIDENT_CATS,
        LENGTH_INCIDENT_CATS,
        DENSITY_CATS,
        CONFLICT_CATS,
        REDLIGHTS_CATS,
        WORKLOAD_CATS,
        PEAK_CATS,
    ]

    for cat_dict, radio_key in zip(
        all_cat_dicts,
        [g[2] for g in CATEGORY_GROUPS]
    ):

        selected_label = categories[radio_key]

        feat_mapping = cat_dict[selected_label]

        features.update(feat_mapping)

    for i in range(1, 16):

        key = f"feature_{i}"

        if key not in features:
            features[key] = False

    return features


# ======================
# 特征字典转换为可读描述
# ======================
def features_to_description(features):

    descs = []

    # number_incident
    if features["feature_1"]:
        descs.append("number_incident = 0")
    else:
        descs.append("number_incident > 0")

    # length_incident
    if features["feature_2"]:
        descs.append("length_incident = 0")
    elif features["feature_3"]:
        descs.append("0 < length_incident ≤ 3")
    else:
        descs.append("length_incident > 3")

    # density
    if features["feature_4"]:
        descs.append("density ≤ 0.15")
    elif features["feature_5"]:
        descs.append("0.15 < density ≤ 0.28")
    elif features["feature_6"]:
        descs.append("0.28 < density ≤ 0.45")
    else:
        descs.append("density > 0.45")

    # conflict
    if features["feature_7"]:
        descs.append("conflict = 0")
    elif features["feature_8"]:
        descs.append("0 < conflict ≤ 0.25")
    else:
        descs.append("conflict > 0.25")

    # redlights
    if features["feature_9"]:
        descs.append("redlights = 0")
    elif features["feature_10"]:
        descs.append("0 < redlights ≤ 0.25")
    else:
        descs.append("redlights > 0.25")

    # workload
    if features["feature_11"]:
        descs.append("zonehour_typical_workload ≤ 1")
    elif features["feature_12"]:
        descs.append("1 < zonehour_typical_workload ≤ 2")
    elif features["feature_13"]:
        descs.append("2 < zonehour_typical_workload ≤ 3")
    elif features["feature_14"]:
        descs.append("3 < zonehour_typical_workload ≤ 4")
    else:
        descs.append("zonehour_typical_workload > 4")

    # peak
    if features["feature_15"]:
        descs.append("peak = 1")
    else:
        descs.append("peak = 0")

    return descs


# ======================
# 规则校验
# ======================
def is_feasible(f):

    fv = {
        k: (1 if v else 0)
        for k, v in f.items()
    }

    if fv["feature_2"] == 1 and fv["feature_3"] != 1:
        return False

    if fv["feature_4"] == 1 and (
        fv["feature_5"] != 1 or fv["feature_6"] != 1
    ):
        return False

    if fv["feature_5"] == 1 and fv["feature_6"] != 1:
        return False

    if fv["feature_7"] == 1 and fv["feature_8"] != 1:
        return False

    if fv["feature_9"] == 1 and fv["feature_10"] != 1:
        return False

    if fv["feature_11"] == 1 and (
        fv["feature_12"] != 1
        or fv["feature_13"] != 1
        or fv["feature_14"] != 1
    ):
        return False

    if fv["feature_12"] == 1 and (
        fv["feature_13"] != 1
        or fv["feature_14"] != 1
    ):
        return False

    if fv["feature_13"] == 1 and fv["feature_14"] != 1:
        return False

    return True


# ======================
# 中间指针式仪表盘
#
# 关键修改：
# 1. 使用 gauge+number
# 2. 数字位置由 Plotly 自动控制
# 3. 字体大小与左右两侧一致：25
# 4. 删除 add_annotation()
# 5. 只用 add_shape() 叠加蓝色指针
# ======================
def make_pointer_gauge(
    value,
    gauge_min=-100,
    gauge_max=100,
    height=GAUGE_HEIGHT
):

    value = float(value)

    value_fraction = (
        (value - gauge_min) / (gauge_max - gauge_min)
    )

    value_fraction = max(
        0.0,
        min(1.0, value_fraction)
    )

    needle_angle = np.pi * (1 - value_fraction)

    tick_values = np.linspace(
        gauge_min,
        gauge_max,
        5
    )

    fig = go.Figure(go.Indicator(

        # 与左右仪表盘采用相同显示模式
        mode="gauge+number",

        value=value,

        # 数字设置
        # 与左右两侧保持相同字体大小
        number={
            "suffix": " seconds per train section",
            "valueformat": ".2f",
            "font": {
                "size": 25,
                "color": "#8492a6"
            }
        },

        gauge={

            "shape": "angular",

            "axis": {
                "range": [
                    gauge_min,
                    gauge_max
                ],

                "tickvals": tick_values,

                "ticktext": [
                    f"{v:.0f}"
                    for v in tick_values
                ],

                "tickfont": {
                    "size": 16
                },

                "showticklabels": True,
            },

            # 不显示仪表盘液柱
            "bar": {
                "color": "rgba(0,0,0,0)",
                "thickness": 0
            },
        },
    ))

    # ======================
    # 蓝色指针坐标
    # ======================

    center_x = 0.5
    center_y = 0.32

    radius_x = 0.24
    radius_y = 0.34

    # 指针箭头尖端
    tip_x = (
        center_x
        + radius_x * np.cos(needle_angle)
    )

    tip_y = (
        center_y
        + radius_y * np.sin(needle_angle)
    )

    # 指针尾部
    # 根据数字自动定位后的位置进行微调
    tail_x = 0.5
    tail_y = 0.20

    # 箭头尺寸
    head_length = 0.045
    head_width = 0.018

    base_x = (
        tip_x
        - head_length * np.cos(needle_angle)
    )

    base_y = (
        tip_y
        - head_length * np.sin(needle_angle)
    )

    perpendicular_x = -np.sin(needle_angle)
    perpendicular_y = np.cos(needle_angle)

    # ======================
    # 指针主体
    # ======================
    fig.add_shape(
        type="path",

        path=(
            f"M {tail_x:.6f},{tail_y:.6f} "
            f"L {tip_x:.6f},{tip_y:.6f}"
        ),

        line={
            "color": "#2563eb",
            "width": 3
        },

        layer="above",

        xref="paper",
        yref="paper",
    )

    # ======================
    # 指针箭头
    # ======================
    fig.add_shape(
        type="path",

        path=(
            f"M {tip_x:.6f},{tip_y:.6f} "

            f"L {base_x + head_width * perpendicular_x:.6f},"
            f"{base_y + head_width * perpendicular_y:.6f} "

            f"L {base_x - head_width * perpendicular_x:.6f},"
            f"{base_y - head_width * perpendicular_y:.6f} Z"
        ),

        fillcolor="#2563eb",

        line={
            "color": "#2563eb",
            "width": 1
        },

        layer="above",

        xref="paper",
        yref="paper",
    )

    # ======================
    # 布局
    # ======================
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ======================
# 分析与结果展示
# ======================
if analyze:

    selected_features = categories_to_features(
        selected_categories
    )

    # ======================
    # 显示当前特征组合
    # ======================
    desc_list = features_to_description(
        selected_features
    )

    desc_text = "; ".join(desc_list)

    st.markdown(
        f"""
        <div style="
            background:#f3f4f6;
            padding:12px;
            border-radius:6px;
            margin-top:10px;
            margin-bottom:10px;
        ">
        <b>Current feature combination:</b>
        {desc_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======================
    # 可行性校验
    # ======================
    feasible = is_feasible(selected_features)

    tree_used_ratio = None

    if feasible:

        match_mask = pd.Series(
            [True] * len(df_features)
        )

        for k, v in selected_features.items():

            match_mask &= (
                df_features[k] == (1 if v else 0)
            )

        matched_ids = df_features.loc[
            match_mask,
            "id"
        ]

        if len(matched_ids) == 0:

            st.error(
                "No matching feature combination found in the dataset."
            )

            mean_val = 0.0
            pos_ratio = 0.0
            neg_ratio = 0.0

        else:

            target_id = str(
                matched_ids.iloc[0]
            )

            df_stats[stats_id_col] = (
                df_stats[stats_id_col].astype(str)
            )

            stat_row = df_stats[
                df_stats[stats_id_col] == target_id
            ]

            if len(stat_row) == 0:

                st.error(
                    "Pre-computed statistics not found for this combination."
                )

                mean_val = 0.0
                pos_ratio = 0.0
                neg_ratio = 0.0

            else:

                row = stat_row.iloc[0]

                mean_val = float(row["mean"])

                pos_ratio = float(
                    row["greater_0"]
                )

                neg_ratio = float(
                    row["less_0"]
                )

        # ======================
        # 查询决策树比例
        # ======================
        webapp2_mask = pd.Series(
            True,
            index=df_webapp2.index
        )

        for k, v in selected_features.items():

            webapp2_mask &= (
                df_webapp2[k].astype(int)
                == (1 if v else 0)
            )

        matched_ratios = df_webapp2.loc[
            webapp2_mask,
            "tree_used_ratio"
        ]

        if len(matched_ratios) > 0:

            raw_ratio = (
                str(matched_ratios.iloc[0])
                .replace("%", "")
                .strip()
            )

            try:

                parsed_ratio = float(raw_ratio)

                if pd.notna(parsed_ratio):
                    tree_used_ratio = parsed_ratio

            except (TypeError, ValueError):
                pass

    else:

        mean_val = 0.0
        pos_ratio = 0.0
        neg_ratio = 0.0

    # ======================
    # 三个仪表盘
    # ======================

    gauge_min = -100
    gauge_max = 100

    g1, g2, g3 = st.columns(3)

    # ======================
    # G1：预测覆盖对列车准点率造成损害的概率
    # ======================
    with g1:

        st.markdown(
            "<div class='gauge-title'>"
            "Predicted probability that override harms train punctuality"
            "</div>",
            unsafe_allow_html=True,
        )

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=float(pos_ratio) * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 25
                }
            },

            gauge={

                "shape": "angular",

                "axis": {
                    "range": [0, 100],
                    "tickfont": {
                        "size": 16
                    },
                    "showticklabels": True
                },

                "bar": {
                    "color": "#dc2626",
                    "thickness": 0.25
                }
            }
        ))

        fig.update_layout(
            height=GAUGE_HEIGHT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ======================
    # G2：预测覆盖后列车延误变化
    # ======================
    with g2:

        st.markdown(
            "<div class='gauge-title'>"
            "Predicted change in train delay if overridden"
            "</div>",
            unsafe_allow_html=True,
        )

        fig = make_pointer_gauge(
            mean_val,
            gauge_min=gauge_min,
            gauge_max=gauge_max,
            height=GAUGE_HEIGHT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ======================
    # G3：预测覆盖改善列车准点率的概率
    # ======================
    with g3:

        st.markdown(
            "<div class='gauge-title'>"
            "Predicted probability that override improves train punctuality"
            "</div>",
            unsafe_allow_html=True,
        )

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=float(neg_ratio) * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 25
                }
            },

            gauge={

                "shape": "angular",

                "axis": {
                    "range": [0, 100],
                    "tickfont": {
                        "size": 16
                    },
                    "showticklabels": True
                },

                "bar": {
                    "color": "#059669",
                    "thickness": 0.25
                }
            }
        ))

        fig.update_layout(
            height=GAUGE_HEIGHT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ======================
    # 决策树比例
    # ======================
    if feasible:

        if tree_used_ratio is None:

            ratio_text = "N/A"

        else:

            tree_used_pct = (
                tree_used_ratio * 100
                if abs(tree_used_ratio) <= 1
                else tree_used_ratio
            )

            ratio_text = f"{tree_used_pct:.2f}%"

        st.markdown(
            "<div style='text-align:center;font-size:28px;'>"
            "<b>Percentage of retained decision trees used to generate "
            "decision-support recommendations: "
            f"{ratio_text}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div style="
                text-align:center;
                color:red;
                font-size:28px;
            ">
            <b>
            Infeasible scenario
            (Decision support not provided)
            </b>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("---")
