import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


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

    def greenhouse_category(score):
        if score <= 4:
            return "Low"
        elif score <= 7:
            return "Moderate"
        else:
            return "High"

    df["Greenhouse Category"] = df["Greenhouse Gas Score"].apply(greenhouse_category)
    
    st.write ("This model predicts whether a vehicle has a Low, Moderate, or High greenhouse gas performance category. The original Greenhouse Gas Score ranges from 1 to 10, but we grouped the scores into three categories: Low = 1-4, Moderate = 5-7, and High = 8-10.")
    

    # Display the category distribution
    st.subheader("Category Distribution")
    category_counts = df["Greenhouse Category"].value_counts().reindex(["Low", "Moderate", "High"]).reset_index()
    category_counts.columns = ["Greenhouse Category", "Count"]
    st.dataframe(category_counts, use_container_width=True, hide_index=True)
    st.divider()


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
    categorical_cols = model_df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
    X = model_df

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Greenhouse Category"])


    # Let user control
    st.subheader("Select Features to Train Model")

    col1, col2, = st.columns([3,1])
    with col1:
        all_features = list(X.columns)
        selected_features = st.multiselect("Select explanatory variables", options=all_features, default=all_features)
    with col2:
        test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20)

    # Train model
    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=(test_size/100), random_state=42, stratify=y)
    
    st.write(f"Training set: {len(X_train):,} samples &nbsp;&nbsp;|&nbsp;&nbsp;"
             f"Test set: {len(X_test):,} samples")
    train_button = st.button("🚀 Train Decision Tree Model", use_container_width=True)

    if train_button:
        if len(selected_features) == 0:
            st.warning("⚠️ Please select at least one explanatory variable.")

        else:   
            # Create Decision Tree    
            tree = DecisionTreeClassifier(max_depth=4, random_state=42)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)

            st.subheader("🎯 Model Performace")
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {accuracy:.2%}")

            labels = target_encoder.classes_


            # Classification report
            st.subheader("📝 Classification Report")
            report = metrics.classification_report(y_test, y_pred, target_names=labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)


            # Confusion matrix
            st.subheader("🤔 Confusion Matrix")
            cm = metrics.confusion_matrix(y_test, y_pred, labels=target_encoder.transform(["Low", "Moderate", "High"]))
            fig, ax = plt.subplots(figsize = (6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Moderate", "High"], yticklabels=["Low", "Moderate", "High"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)


            # Feature Importance
            st.subheader("📌 Top Feature Importances")
            importance_df = pd.DataFrame({"Feature": X_selected.columns, "Importance": tree.feature_importances_}).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize = (7,5))
            ax2.barh(importance_df["Feature"], importance_df["Importance"])
            ax2.set_xlabel("Importance")
            ax2.set_ylabel("Feature")
            ax2.set_title("Top Feature Importances")
            st.pyplot(fig2)
