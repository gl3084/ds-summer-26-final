import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

def render():
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #16a34a;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    
    </style>
    """, unsafe_allow_html=True)


    st.title("🌳 Decision Tree Model")

    df_withelectric = pd.read_excel("CO2.xlsx")
    df = df_withelectric[df_withelectric["Fuel"] != "Electricity"]
    df = df[df["Fuel"] != "Hydrogen"]
    df = df[df["Fuel"] != "Electricity/Hydrogen"]


    # CONDENCING EACH ENTRY BY BRAND
    multi_word_brands = [
        "ALFA ROMEO",
        "ASTON MARTIN",
        "LAND ROVER",
        "ROLLS ROYCE",
        "MERCEDES-BENZ",
    ]

    def get_manufacturer(Model):
        Model = str(Model)
        for brand in multi_word_brands:
            if Model.startswith(brand):
                return brand
        return Model.split()[0]

    df["Manufacturer"] = df["Model"].apply(get_manufacturer)


    def greenhouse_category(score):
        if score <= 4:
            return "Low"
        elif score <= 7:
            return "Moderate"
        else:
            return "High"

    df["Greenhouse Category"] = df["Greenhouse Gas Score"].apply(greenhouse_category)


    # Keep gasoline MPG value for fuel flex/hybrid cars and convert to integers
    df["City MPG"] = df["City MPG"].str.split("/").str[0]
    df["Hwy MPG"] = df["Hwy MPG"].str.split("/").str[0]
    df["Cmb MPG"] = df["Cmb MPG"].str.split("/").str[0]

    df["City MPG"] = pd.to_numeric(df["City MPG"])
    df["Hwy MPG"] = pd.to_numeric(df["Hwy MPG"])
    df["Cmb MPG"] = pd.to_numeric(df["Cmb MPG"])


    # Drop target/leakage/not useful predictive features for the model
    drop_cols = ["Model", 
                 "Underhood ID", 
                 "Stnd", 
                 "Stnd Description", 
                 "Comb CO2", 
                 "Greenhouse Gas Score", 
                 "Greenhouse Category", 
                 "SmartWay", 
                 "Air Pollution Score"]
    model_df = df.drop(columns=drop_cols)


    # Encode categorical columns
    encoders = {}
    categorical_cols = model_df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le
    X = model_df

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Greenhouse Category"])


    # Let user control
    st.subheader("👇 Select Features to Train Model")

    col1, col2, = st.columns([3,1])
    with col1:
        all_features = list(X.columns)
        selected_features = st.multiselect("Select explanatory variables", options=all_features, default=all_features)
    with col2:
        test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20)


    # Train model
    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=(test_size/100), random_state=42, stratify=y)


    # THIS IS WHERE PARAMETERS CAN BE CHANGED (ADD USER INPUT BEFORE SO THAT YOU CAN DO: tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, etc.))
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)


    st.subheader("🎯 Model Performace")
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {accuracy:.2%}")