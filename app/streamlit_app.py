"""
Study Crash Predictor - Streamlit Web App.

A polished, modern interface for predicting study focus states
and receiving smart recommendations.
"""

import os
import sys
import json
import joblib
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import load_artifacts, predict_single, get_recommendation
from src.train_models import FEATURE_COLS

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Study Crash Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - polished dark-academic theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Main background ---- */
    .stApp {
        background: #0e1117;
        color: #e6edf3;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }

    section[data-testid="stSidebar"] h2 {
        color: #f0f6fc !important;
        font-weight: 700;
        font-size: 1.3rem;
        letter-spacing: -0.02em;
    }

    /* ---- Sidebar slider styling ---- */
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        color: #58a6ff !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [aria-label] {
        background: #21262d !important;
    }

    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTrack"] {
        background: linear-gradient(90deg, #238636, #58a6ff) !important;
    }

    /* ---- Header ---- */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        position: relative;
    }

    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #f778ba 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }

    .hero-header p {
        color: #8b949e;
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.3);
        color: #58a6ff;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #c9d1d9;
    }

    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
        background: transparent !important;
    }

    /* ---- Result Card ---- */
    .result-card {
        padding: 2.2rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.15;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.15'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }

    .result-card h2 {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }

    .result-card .subtitle {
        font-size: 1rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }

    .result-card .confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 800;
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }

    .result-focus {
        background: linear-gradient(135deg, #0d4429 0%, #1a7f4e 50%, #2ea043 100%);
        border: 1px solid rgba(46,160,67,0.4);
        box-shadow: 0 0 40px rgba(46,160,67,0.15);
    }

    .result-distracted {
        background: linear-gradient(135deg, #5c2d0e 0%, #9e6a03 50%, #d29922 100%);
        border: 1px solid rgba(210,153,34,0.4);
        box-shadow: 0 0 40px rgba(210,153,34,0.15);
    }

    .result-crash {
        background: linear-gradient(135deg, #5c1d16 0%, #a32720 50%, #da3633 100%);
        border: 1px solid rgba(218,54,51,0.4);
        box-shadow: 0 0 40px rgba(218,54,51,0.15);
    }

    /* ---- Recommendation Box ---- */
    .recommendation-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        padding: 1.5rem 1.8rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
        color: #c9d1d9;
        line-height: 1.8;
    }

    .recommendation-box h3 {
        color: #f0f6fc;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        font-weight: 700;
    }

    .recommendation-box strong {
        color: #58a6ff;
    }

    /* ---- Glass card ---- */
    .glass-card {
        background: rgba(22,27,34,0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* ---- Metric cards ---- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 1rem 0;
    }

    .metric-item {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-item:hover {
        border-color: #58a6ff;
    }

    .metric-item .label {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .metric-item .value {
        color: #f0f6fc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* ---- Status indicators ---- */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s infinite;
    }

    .status-dot.green { background: #2ea043; box-shadow: 0 0 8px #2ea043; }
    .status-dot.yellow { background: #d29922; box-shadow: 0 0 8px #d29922; }
    .status-dot.red { background: #da3633; box-shadow: 0 0 8px #da3633; }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ---- Preset buttons ---- */
    .preset-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem 0;
    }

    .preset-btn {
        background: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
    }

    .preset-btn:hover {
        background: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* ---- StButton ---- */
    .stButton button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: 1px solid rgba(46,160,67,0.5);
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 10px;
        width: 100%;
        transition: all 0.2s;
        letter-spacing: -0.01em;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        border-color: #3fb950;
        box-shadow: 0 0 20px rgba(46,160,67,0.3);
        transform: translateY(-1px);
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        padding: 3rem 0 1.5rem 0;
        color: #484f58;
        font-size: 0.85rem;
        border-top: 1px solid #21262d;
        margin-top: 2rem;
    }

    .app-footer a {
        color: #58a6ff;
        text-decoration: none;
    }

    /* ---- Hide streamlit elements ---- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Probability gauge ---- */
    .gauge-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }

    /* ---- Section headers ---- */
    .section-header {
        color: #f0f6fc;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header .icon {
        font-size: 1.2rem;
    }

    /* ---- Divider ---- */
    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 2rem 0;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROB_COLORS = {
    "Deep Focus": "#2ea043",
    "Distracted": "#d29922",
    "Study Crash Incoming": "#da3633",
}

PROB_ICONS = {
    "Deep Focus": "",
    "Distracted": "",
    "Study Crash Incoming": "",
}

CARD_CLASS = {
    "Deep Focus": "result-focus",
    "Distracted": "result-distracted",
    "Study Crash Incoming": "result-crash",
}

PRESETS = {
    "Perfect Session": {
        "hours_slept": 8.0, "study_duration_minutes": 30,
        "phone_pickups_last_hour": 1, "social_media_opens_last_hour": 0,
        "break_duration_minutes": 15, "subject_difficulty": 4,
        "current_time_of_day": 10, "caffeine_intake": 1,
        "previous_day_productivity": 8.5, "mood_level": 9.0,
    },
    "Getting Distracted": {
        "hours_slept": 6.5, "study_duration_minutes": 75,
        "phone_pickups_last_hour": 8, "social_media_opens_last_hour": 5,
        "break_duration_minutes": 5, "subject_difficulty": 6,
        "current_time_of_day": 15, "caffeine_intake": 2,
        "previous_day_productivity": 5.0, "mood_level": 5.5,
    },
    "About to Crash": {
        "hours_slept": 5.0, "study_duration_minutes": 180,
        "phone_pickups_last_hour": 15, "social_media_opens_last_hour": 10,
        "break_duration_minutes": 2, "subject_difficulty": 9,
        "current_time_of_day": 23, "caffeine_intake": 4,
        "previous_day_productivity": 3.0, "mood_level": 2.5,
    },
}

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "hours_slept": 7.0,
    "study_duration_minutes": 45,
    "phone_pickups_last_hour": 3,
    "social_media_opens_last_hour": 2,
    "break_duration_minutes": 10,
    "subject_difficulty": 5,
    "current_time_of_day": 14,
    "caffeine_intake": 1,
    "previous_day_productivity": 6.0,
    "mood_level": 7.0,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def apply_preset(preset_name: str):
    for k, v in PRESETS[preset_name].items():
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-badge">Machine Learning Powered</div>
        <h1>Study Crash Predictor</h1>
        <p>Real-time analysis of your study habits to predict focus state,
        detect distractions, and warn you before burnout hits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - Input Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Study Session Inputs")

# Presets
st.sidebar.markdown("**Quick Presets**")
preset_cols = st.sidebar.columns(3)
for i, name in enumerate(PRESETS.keys()):
    if preset_cols[i].button(name, key=f"preset_{i}", use_container_width=True):
        apply_preset(name)
        st.rerun()

st.sidebar.markdown("---")

# Sleep & Breaks section
st.sidebar.markdown("### Sleep & Breaks")
hours_slept = st.sidebar.slider(
    "Hours Slept Last Night", 4.0, 10.0, key="hours_slept", step=0.5,
    help="How many hours did you sleep last night?",
)
break_duration = st.sidebar.slider(
    "Last Break Duration (min)", 0, 60, key="break_duration_minutes",
    help="How long was your most recent break?",
)

# Session section
st.sidebar.markdown("### Session Info")
study_duration = st.sidebar.slider(
    "Study Duration (min)", 10, 240, key="study_duration_minutes", step=5,
    help="How long have you been studying in this session?",
)
time_of_day = st.sidebar.slider(
    "Current Hour (24h)", 6, 23, key="current_time_of_day",
    help="What hour is it right now?",
)
subject_difficulty = st.sidebar.slider(
    "Subject Difficulty (1-10)", 1, 10, key="subject_difficulty",
    help="How hard is the material you're studying?",
)

# Phone & Digital section
st.sidebar.markdown("### Digital Activity")
phone_pickups = st.sidebar.slider(
    "Phone Pickups (last hour)", 0, 30, key="phone_pickups_last_hour",
    help="Times you picked up your phone in the last 60 minutes.",
)
social_media = st.sidebar.slider(
    "Social Media Opens (last hour)", 0, 20, key="social_media_opens_last_hour",
    help="Times you opened social media apps.",
)

# Wellbeing section
st.sidebar.markdown("### Wellbeing")
caffeine = st.sidebar.slider(
    "Caffeine (cups today)", 0, 6, key="caffeine_intake",
    help="Coffee or tea cups consumed today.",
)
prev_productivity = st.sidebar.slider(
    "Yesterday's Productivity (1-10)", 1.0, 10.0, key="previous_day_productivity", step=0.5,
    help="How productive were you yesterday overall?",
)
mood = st.sidebar.slider(
    "Current Mood (1-10)", 1.0, 10.0, key="mood_level", step=0.5,
    help="Rate your current mood and energy level.",
)

# Build feature dictionary
features = {
    "hours_slept": st.session_state.hours_slept,
    "study_duration_minutes": st.session_state.study_duration_minutes,
    "phone_pickups_last_hour": st.session_state.phone_pickups_last_hour,
    "social_media_opens_last_hour": st.session_state.social_media_opens_last_hour,
    "break_duration_minutes": st.session_state.break_duration_minutes,
    "subject_difficulty": st.session_state.subject_difficulty,
    "current_time_of_day": st.session_state.current_time_of_day,
    "caffeine_intake": st.session_state.caffeine_intake,
    "previous_day_productivity": st.session_state.previous_day_productivity,
    "mood_level": st.session_state.mood_level,
}

# ---------------------------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_insights, tab_about = st.tabs(["Prediction", "Model Insights", "About"])

# ========================
# TAB 1: PREDICTION
# ========================
with tab_predict:

    # Input summary strip
    st.markdown('<div class="section-header"><span class="icon"></span> Current Inputs</div>', unsafe_allow_html=True)
    summary_items = [
        ("Sleep", f"{features['hours_slept']}h"),
        ("Session", f"{features['study_duration_minutes']}m"),
        ("Phone", str(features['phone_pickups_last_hour'])),
        ("Social", str(features['social_media_opens_last_hour'])),
        ("Break", f"{features['break_duration_minutes']}m"),
        ("Difficulty", f"{features['subject_difficulty']}/10"),
        ("Hour", f"{features['current_time_of_day']}:00"),
        ("Caffeine", f"{features['caffeine_intake']} cups"),
        ("Prod.", f"{features['previous_day_productivity']}/10"),
        ("Mood", f"{features['mood_level']}/10"),
    ]

    metric_html = '<div class="metric-grid">'
    for label, value in summary_items:
        metric_html += f'<div class="metric-item"><div class="label">{label}</div><div class="value">{value}</div></div>'
    metric_html += '</div>'
    st.markdown(metric_html, unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("Analyze My Study State", use_container_width=True)

    if predict_btn:
        try:
            label, probabilities = predict_single(features)

            # Confidence of the predicted class
            confidence = probabilities.get(label, 0) * 100

            # ---- Result Card ----
            card_css = CARD_CLASS.get(label, "result-focus")
            dot_class = {"Deep Focus": "green", "Distracted": "yellow", "Study Crash Incoming": "red"}.get(label, "green")

            st.markdown(
                f"""
                <div class="result-card {card_css}">
                    <div style="position:relative;z-index:1;">
                        <span class="status-dot {dot_class}"></span>
                        <span style="opacity:0.7;font-size:0.85rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">
                            Predicted State
                        </span>
                    </div>
                    <h2>{label}</h2>
                    <div class="confidence">{confidence:.1f}%</div>
                    <p class="subtitle">model confidence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---- Probability Breakdown ----
            st.markdown('<div class="section-header"><span class="icon"></span> Probability Breakdown</div>', unsafe_allow_html=True)

            prob_cols = st.columns(3)
            for i, (cls, prob) in enumerate(sorted(probabilities.items(), key=lambda x: -x[1])):
                pct = prob * 100
                color = PROB_COLORS.get(cls, "#58a6ff")
                icon = PROB_ICONS.get(cls, "")

                with prob_cols[i]:
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:1rem; background:#161b22; border:1px solid #30363d; border-radius:10px;">
                            <div style="font-size:1.5rem;">{icon}</div>
                            <div style="color:{color}; font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:800; margin:0.3rem 0;">{pct:.1f}%</div>
                            <div style="color:#8b949e; font-size:0.8rem; font-weight:600;">{cls}</div>
                            <div style="background:#21262d; border-radius:4px; height:6px; margin-top:0.7rem; overflow:hidden;">
                                <div style="background:{color}; width:{pct}%; height:100%; border-radius:4px; transition:width 0.5s;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ---- Probability Bar Chart (matplotlib) ----
            st.markdown("")
            fig, ax = plt.subplots(figsize=(10, 2.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1])
            classes = [p[0] for p in sorted_probs]
            vals = [p[1] * 100 for p in sorted_probs]
            colors = [PROB_COLORS.get(c, "#58a6ff") for c in classes]

            bars = ax.barh(classes, vals, color=colors, height=0.55, edgecolor="none")
            ax.set_xlim(0, 105)
            ax.tick_params(axis="y", colors="#8b949e", labelsize=10)
            ax.tick_params(axis="x", colors="#484f58", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#21262d")
            ax.spines["left"].set_color("#21262d")
            ax.xaxis.grid(True, color="#21262d", linestyle="--", alpha=0.5)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=11, fontweight="bold",
                        color="#c9d1d9", fontfamily="JetBrains Mono")

            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ---- Recommendations ----
            st.markdown('<div class="section-header"><span class="icon"></span> Smart Recommendations</div>', unsafe_allow_html=True)
            recommendation = get_recommendation(label)
            st.markdown(
                f"""
                <div class="recommendation-box">
                    {recommendation.replace(chr(10), "<br>")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except FileNotFoundError:
            st.error(
                "Model not found. Run the training pipeline first:\n\n"
                "```bash\n"
                "python -m src.generate_data\n"
                "python -m src.train_models\n"
                "python -m src.evaluate\n"
                "```"
            )

    else:
        # Placeholder state
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 2rem; color:#484f58;">
                <div style="font-size:4rem; margin-bottom:1rem;"></div>
                <h3 style="color:#8b949e; font-weight:600; margin-bottom:0.5rem;">Ready to Analyze</h3>
                <p>Adjust the inputs in the sidebar and click <strong style="color:#2ea043;">Analyze My Study State</strong> to get your prediction.</p>
                <p style="font-size:0.85rem; margin-top:1rem;">
                    Or try a <strong>Quick Preset</strong> in the sidebar to see example results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ========================
# TAB 2: MODEL INSIGHTS
# ========================
with tab_insights:

    st.markdown('<div class="section-header"><span class="icon"></span> Model Performance</div>', unsafe_allow_html=True)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "plots")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Model comparison cards from JSON
    comp_path = os.path.join(models_dir, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)

        sorted_models = sorted(comparison.items(), key=lambda x: -x[1]["f1_score"])
        cols = st.columns(len(sorted_models))

        for i, (name, metrics) in enumerate(sorted_models):
            is_best = i == 0
            border_color = "#2ea043" if is_best else "#30363d"
            badge = '<span style="background:#238636;color:white;padding:2px 8px;border-radius:4px;font-size:0.65rem;font-weight:700;margin-left:6px;">BEST</span>' if is_best else ""

            with cols[i]:
                st.markdown(
                    f"""
                    <div style="background:#161b22; border:1px solid {border_color}; border-radius:12px; padding:1.2rem; text-align:center;">
                        <div style="color:#f0f6fc; font-weight:700; font-size:0.95rem; margin-bottom:0.8rem;">
                            {name}{badge}
                        </div>
                        <div style="color:#58a6ff; font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:800;">
                            {metrics['accuracy']*100:.1f}%
                        </div>
                        <div style="color:#8b949e; font-size:0.75rem; margin-bottom:0.8rem;">Accuracy</div>
                        <div style="display:flex; justify-content:space-around; border-top:1px solid #21262d; padding-top:0.8rem;">
                            <div>
                                <div style="color:#c9d1d9; font-weight:700; font-family:'JetBrains Mono',monospace;">{metrics['f1_score']:.3f}</div>
                                <div style="color:#484f58; font-size:0.7rem;">F1</div>
                            </div>
                            <div>
                                <div style="color:#c9d1d9; font-weight:700; font-family:'JetBrains Mono',monospace;">{metrics['cv_mean']:.3f}</div>
                                <div style="color:#484f58; font-size:0.7rem;">CV Mean</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Charts in a 2x2 grid
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fi_path = os.path.join(plots_dir, "feature_importance.png")
        if os.path.exists(fi_path):
            st.markdown('<div class="section-header"><span class="icon"></span> Feature Importance</div>', unsafe_allow_html=True)
            st.image(fi_path, use_container_width=True)

    with chart_col2:
        mc_path = os.path.join(plots_dir, "model_comparison.png")
        if os.path.exists(mc_path):
            st.markdown('<div class="section-header"><span class="icon"></span> Accuracy vs F1</div>', unsafe_allow_html=True)
            st.image(mc_path, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        cv_path = os.path.join(plots_dir, "cross_validation.png")
        if os.path.exists(cv_path):
            st.markdown('<div class="section-header"><span class="icon"></span> Cross-Validation</div>', unsafe_allow_html=True)
            st.image(cv_path, use_container_width=True)

    with chart_col4:
        cm_path = os.path.join(plots_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.markdown('<div class="section-header"><span class="icon"></span> Confusion Matrices</div>', unsafe_allow_html=True)
            st.image(cm_path, use_container_width=True)


# ========================
# TAB 3: ABOUT
# ========================
with tab_about:

    st.markdown('<div class="section-header"><span class="icon"></span> About This Project</div>', unsafe_allow_html=True)

    about_col1, about_col2 = st.columns([2, 1])

    with about_col1:
        st.markdown(
            """
            **Study Crash Predictor** is a machine learning application that analyzes
            real-time behavioral signals to predict a student's current study state.

            ### How It Works

            1. **Input** - Enter your current study habits and daily behavior through the sidebar sliders.
            2. **Predict** - The model processes 10 features through a trained classifier.
            3. **Act** - Receive a state prediction with confidence score and actionable recommendations.

            ### Three Predicted States

            | State | Meaning | Action |
            |---|---|---|
            | **Deep Focus** | You're fully engaged and productive | Keep going, protect your flow |
            | **Distracted** | Attention is slipping | Take corrective action now |
            | **Study Crash Incoming** | Burnout is imminent | Rest, reset, and recover |

            ### Features Used (10 Total)

            | Feature | Why It Matters |
            |---|---|
            | Hours Slept | Sleep deprivation is the #1 crash predictor |
            | Study Duration | Diminishing returns after ~90 minutes |
            | Phone Pickups | Direct measure of digital distraction |
            | Social Media Opens | Attention fragmentation indicator |
            | Break Duration | Recovery time affects sustainability |
            | Subject Difficulty | Harder material drains focus faster |
            | Time of Day | Circadian rhythm affects cognition |
            | Caffeine Intake | Short-term boost, long-term dependency |
            | Previous Productivity | Momentum and habit patterns |
            | Mood Level | Emotional state influences focus |
            """
        )

    with about_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#f0f6fc; margin-bottom:1rem;">Tech Stack</h4>
                <div style="display:flex; flex-direction:column; gap:0.5rem;">
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         Python 3.9+
                    </div>
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         scikit-learn
                    </div>
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         pandas / numpy
                    </div>
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         matplotlib / seaborn
                    </div>
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         Streamlit
                    </div>
                    <div style="background:#21262d; padding:0.5rem 0.8rem; border-radius:6px; color:#c9d1d9; font-size:0.85rem;">
                         joblib
                    </div>
                </div>
            </div>

            <div class="glass-card" style="margin-top:1rem;">
                <h4 style="color:#f0f6fc; margin-bottom:1rem;">Models Compared</h4>
                <div style="display:flex; flex-direction:column; gap:0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; background:#21262d; padding:0.5rem 0.8rem; border-radius:6px;">
                        <span style="color:#c9d1d9; font-size:0.85rem;">Random Forest</span>
                        <span style="color:#2ea043; font-size:0.75rem; font-weight:600;">Ensemble</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:#21262d; padding:0.5rem 0.8rem; border-radius:6px;">
                        <span style="color:#c9d1d9; font-size:0.85rem;">Gradient Boosting</span>
                        <span style="color:#2ea043; font-size:0.75rem; font-weight:600;">Ensemble</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:#21262d; padding:0.5rem 0.8rem; border-radius:6px;">
                        <span style="color:#c9d1d9; font-size:0.85rem;">Logistic Regression</span>
                        <span style="color:#58a6ff; font-size:0.75rem; font-weight:600;">Linear</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:#21262d; padding:0.5rem 0.8rem; border-radius:6px;">
                        <span style="color:#c9d1d9; font-size:0.85rem;">SVM</span>
                        <span style="color:#bc8cff; font-size:0.75rem; font-weight:600;">Kernel</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <p>Built with Streamlit & scikit-learn | <strong>Study Crash Predictor</strong></p>
        <p style="margin-top:0.3rem;">
            A machine learning project for predicting student focus states
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
