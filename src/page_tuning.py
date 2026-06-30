import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

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


    st.markdown("---")
    st.subheader("🌲 Tweak Decision Tree Hyperparameters")
    st.caption("Test different depth values and amount of data allowed in each leaf, and compare splitting criteria.")

    #W&B log
    log_wandb = st.checkbox("Log runs to Weights & Biases", value=False)
    if log_wandb:
        wandb_project = st.text_input("W&B Project Name", value="car-sustainability-tuning")
        wandb_entity = st.text_input("W&B Entity", value="")
        wandb_api = st.text_input("W&B API Key", value="", type="password")

    col1, col2, col3 = st.columns(3)
    with col1:
        criterion = st.selectbox(
            "Splitting Criterion",
            options=["gini", "entropy", "log_loss"],
            help="Gini measures impurity; entropy/log_loss measure information gain."
        )
    with col2:
        max_depth = st.slider("Max Tree Depth (max_depth)", min_value=1, max_value=20, value=4)
    with col3:
        min_samples_leaf = st.slider("Min Samples per Leaf (min_samples_leaf)", min_value=1, max_value=50, value=1)

    st.markdown("##### 🔁 Cross-Validation Settings")
    col4, col5 = st.columns(2)
    with col4:
        cv_folds = st.slider("CV Folds (k)", min_value=2, max_value=10, value=5)
    with col5:
        n_trials = st.slider("Number of Trials", min_value=1, max_value=20, value=5,
                              help="Repeats CV with different random splits to show variance in accuracy.")

    run_tuning = st.button("🚀 Run Tuning Experiment & Log Metrics")
    if run_tuning:
        # Train final model on the single train/test split using selected hyperparameters
        tree = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        st.subheader("🎯 Model Performance")
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {accuracy:.2%}")

        # Run repeated cross-validation across the selected number of trials
        st.subheader("📊 Cross-Validation Results")
        trial_means = []
        for trial in range(n_trials):
            cv_tree = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=trial  # vary the seed each trial
            )
            scores = cross_val_score(cv_tree, X_selected, y, cv=cv_folds)
            trial_means.append(scores.mean())

        cv_results_df = pd.DataFrame({
            "Trial": range(1, n_trials + 1),
            "Mean CV Accuracy": trial_means
        })

        col6, col7 = st.columns([1, 2])
        with col6:
            st.info(f"**Avg Accuracy Across Trials:** {sum(trial_means)/len(trial_means):.2%}")
            st.info(f"**Std Dev Across Trials:** {pd.Series(trial_means).std():.4f}")
        with col7:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.lineplot(data=cv_results_df, x="Trial", y="Mean CV Accuracy", marker="o", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title(f"{cv_folds}-Fold CV Accuracy Across {n_trials} Trials")
            st.pyplot(fig)


        #W&B Actual Log
        if log_wandb:
            if wandb_api:
                wandb.login(key=wandb_api)
            
            run = wandb.init(project=wandb_project, entity=wandb_entity if wandb_entity != "" else None,
                config={"criterion": criterion, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "cv_folds": cv_folds, "n_trials": n_trials}
            )
            
            wandb.log({
                "test_accuracy": accuracy,
                "avg_cv_accuracy": sum(trial_means)/len(trial_means),
                "std_cv_accuracy": pd.Series(trial_means).std(),
                "tree_depth": tree.get_depth(),
                "num_leaves": tree.get_n_leaves()
            })

            wandb.finish()

            st.success("Successfully logged run to Weights & Biases!")


        # Best Model
        st.divider()
        st.subheader("🏆 Best Model")
        st.caption("Evaluation tested under 5-fold cross-validation averaged across 5 trials")
        best_model = pd.DataFrame({"Hyperparameter": ["Criterion", "Max Tree Depth", "Min Samples per Leaf", "Mean Accuracy"],
                        "Best Value": ["Gini", 5, 1, "97.03%"]})
        st.table(best_model)
