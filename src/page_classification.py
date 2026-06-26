import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def render():
    st.title("This is the classification model page")

    df_withelectric = pd.read_excel("CO2.xlsx")
    df = df_withelectric.dropna()

    def greenhouse_category(score):
        if score <= 4:
            return "Low"
        elif score <= 7:
            return "Moderate"
        else:
            return "High"

    df["Greenhouse Category"] = df["Greenhouse Gas Score"].apply(greenhouse_category)
    st.dataframe(df.sample(30))
    