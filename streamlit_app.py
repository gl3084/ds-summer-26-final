import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Car Sustainability Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* green */
    :root {
        --khaki: #d1c59b;
        --sage: #809784;
        --forest: #4e7157;
        --ivory: #E6E8E7;
        --black: #161f15;
    }
    .stApp > header {background-color: transparent;}
    .stApp {
        background-color: #809784;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4e7157 0%, #161f15 100%);
        width: 400px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 400px !important;
    }
    }
    [data-testid="stSidebar"] * {color: #E6E8E7 !important;}
    [data-testid="stSidebar"] code {
        background: rgba(230,232,231,0.18) !important;
        color: #E6E8E7 !important;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {color: #E6E8E7 !important;}
    .metric-card {
        background: #E6E8E7;
        border-left: 4px solid #4E7157;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card h3 {margin: 0; font-size: 0.85rem; color: #d1c59b;}
    .metric-card p {margin: 0; font-size: 1.6rem; font-weight: 700; color: #4E7157;}
    .hero-banner {
        background: linear-gradient(135deg, #4E7157 0%, #809784 50%, #d1c59b 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {color: #d1c59b; font-size: 2.2rem; margin-bottom: 0.3rem;}
    .hero-banner p {color: #d1c59b; font-size: 1.1rem; opacity: 0.9;}
    div[data-testid="stMetric"] {
        background: #4e7157;
        border: 1px solid #d1c59b;
        border-radius: 10px;
        padding: 12px 16px;
    }

    button[data-baseweb="tab"] {
        color: #4E7157 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #d1c59b !important;
        border-bottom-color: #d1c59b !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation ──────────────────────────────────────────────────────
from src import page_intro, page_visualization, page_prediction, page_classification, page_explainability, page_tuning

PAGES = {
    "🏠 Introduction & Data": page_intro,
    "📊 Data Visualization": page_visualization,
    "🤖 Model Prediction": page_prediction,
    "📦 Classification Model Prediction": page_classification,
    "🔍 Explainability (SHAP)": page_explainability,
    "⚙️ Hyperparameter Tuning": page_tuning,
}

with st.sidebar:
    st.markdown("## Car Sustainability Predictor")
    st.markdown("---")
    selected = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Matthew Hill, Dylan Wu,")
    st.caption("Gareth Liu, Ricky Lim")
    st.markdown("---")
    st.caption("Summer 2026 ")
    st.caption("NYU Principles of Data Science I")


# ── Render selected page ────────────────────────────────────────────
PAGES[selected].render()
