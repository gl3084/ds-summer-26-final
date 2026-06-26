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
    /* NYU purple accent */
    :root {
        --sky-blue: #37ddfa;
        --earth-green: #4bc920;
    }
    .stApp > header {background-color: transparent;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #37ddfa 0%, #32bd04 100%);
    }
    [data-testid="stSidebar"] * {color: white !important;}
    [data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.18) !important;
        color: #ffd966 !important;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {color: white !important;}
    .metric-card {
        background: #127087;
        border-left: 4px solid #57068C;
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
from src import page_intro, page_visualization, page_prediction, page_explainability, page_tuning

PAGES = {
    "🏠 Business Case & Data": page_intro,
    "📊 Data Visualization": page_visualization,
    "🤖 Model Prediction": page_prediction,
    "🔍 Explainability (SHAP)": page_explainability,
    "⚙️ Hyperparameter Tuning": page_tuning,
}

with st.sidebar:
    st.markdown("## 🎓 Car Sustainability Predictor")
    st.markdown("---")
    selected = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Matthew Hill, Dylan Wu,")
    st.caption("Gareth Liu, Ricky Lim")
    st.markdown("---")
    st.caption("Summer 2026 ")
    st.caption("NYU Principles of Data Science I")


# ── Set up dataset ────────────────────────────────────────────
df = pd.read_excel("CO2.xlsx")
st.dataframe(df.head(5))

st.write(df.shape)

st.markdown("##### Missing values")
missing = df.isnull().sum()
st.write(missing)

if missing.sum() == 0:
    st.success("✅ No missing values found")
else:
    st.warning("⚠️ You have missing values")
        
st.markdown("##### 📈 Summary Statistics")
if st.button("Show Original Dataset Stats"):
    st.dataframe(df.describe())


# Removed electric cars which are all the ones with missing values
st.markdown("##### NO ELECTRIC CAR DATASET")
df_noelectric = df.dropna()
st.write(df_noelectric.columns)
st.write(df_noelectric["Greenhouse Gas Score"].value_counts().sort_index())
st.write(df_noelectric.shape)

st.markdown("##### Missing values of Only Gas Cars")
missing = df_noelectric.isnull().sum()
st.write(missing)

if missing.sum() == 0:
    st.success("✅ No missing values found")
else:
    st.warning("⚠️ You have missing values")
        
st.markdown("##### 📈 Summary Statistics of only Gas Cars")
if st.button("Show Updated Dataset Stats"):
    st.dataframe(df_noelectric.describe())

# ── Render selected page ────────────────────────────────────────────
PAGES[selected].render()