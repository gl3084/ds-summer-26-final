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
    st.title("This is the classification model page")

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

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    st.subheader("Model Performace")
    scores = cross_val_score(tree, X, y, cv=5)
    st.write("Cross-validation scores: ", scores)
    st.success(f"Average accuracy: {scores.mean()}")

    labels = target_encoder.classes_

    # Classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = metrics.confusion_matrix(y_test, y_pred, labels=target_encoder.transform(["Low", "Moderate", "High"]))
    fig, ax = plt.subplots(figsize = (6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Moderate", "High"], yticklabels=["Low", "Moderate", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Top Feature Importances")
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": tree.feature_importances_}).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df, use_container_width=True)
    st.bar_chart(importance_df.set_index("Feature")["Importance"])

    st.write(tree.get_depth())
    st.write(tree.get_n_leaves())
    st.write("Feautre used in model: ", list(X.columns))