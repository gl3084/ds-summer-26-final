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
        --khaki: #ada78a;
        --sage: #809784;
        --forest: #4e7157;
        --ivory: #E6E8E7;
    }
    .stApp > header {background-color: transparent;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ada78a 0%, #4e7157 100%);
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
    .metric-card h3 {margin: 0; font-size: 0.85rem; color: #666;}
    .metric-card p {margin: 0; font-size: 1.6rem; font-weight: 700; color: #57068C;}
    .hero-banner {
        background: linear-gradient(135deg, #57068C 0%, #8900E1 50%, #b347ff 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {color: white; font-size: 2.2rem; margin-bottom: 0.3rem;}
    .hero-banner p {color: #e0c8f0; font-size: 1.1rem;}
    div[data-testid="stMetric"] {
        background: #f8f4fc;
        border: 1px solid #e0d0f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation ──────────────────────────────────────────────────────
from src import page_intro, page_visualization, page_prediction, page_classification, page_explainability, page_tuning

PAGES = {
    "🏠 Business Case & Data": page_intro,
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