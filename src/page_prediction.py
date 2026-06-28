import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXCLUDED_FUELS = ["Electricity", "Hydrogen", "Electricity/Hydrogen"]
DEFAULT_FEATURES = [
    "Displ",
    "Cyl",
    "Trans",
    "Drive",
    "Fuel",
    "Cert Region",
    "Veh Class",
    "Air Pollution Score",
    "City MPG",
    "Hwy MPG",
    "Cmb MPG",
]
TARGET = "Comb CO2"


@st.cache_data
def load_regression_data():
    df = pd.read_excel("CO2.xlsx")
    df = df[~df["Fuel"].isin(EXCLUDED_FUELS)].copy()

    # Plug-in hybrids and flexible-fuel cars store values like "29/77".
    # The first value is the gasoline-side value, which keeps the model
    # comparable with the gas and hybrid classification page.
    for col in ["City MPG", "Hwy MPG", "Cmb MPG", TARGET]:
        df[col] = df[col].astype(str).str.split("/").str[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=DEFAULT_FEATURES + [TARGET]).copy()
    return df


def build_model(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def coefficient_table(model, numeric_features, categorical_features):
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(numeric_features)

    if categorical_features:
        encoded_names = list(
            preprocessor.named_transformers_["categorical"].get_feature_names_out(categorical_features)
        )
        feature_names.extend(encoded_names)

    coefficients = model.named_steps["model"].coef_

    return (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
        .assign(Abs_Coefficient=lambda data: data["Coefficient"].abs())
        .sort_values("Abs_Coefficient", ascending=False)
        .drop(columns="Abs_Coefficient")
        .head(20)
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="regression-metric-card">
            <div class="regression-metric-label">{label}</div>
            <div class="regression-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #16a34a;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            width: 100%;
        }
        .regression-metric-card {
            background: #f8f4fc;
            border: 1px solid #d7c9e8;
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 86px;
        }
        .regression-metric-label {
            color: #4b5563;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .regression-metric-value {
            color: #111827;
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Linear Regression Model: CO2 Emissions")
    st.write(
        "This model predicts combined CO2 emissions (`Comb CO2`) from vehicle specifications "
        "such as engine size, MPG, fuel type, drivetrain, and vehicle class."
    )

    df = load_regression_data()

    st.subheader("Regression Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Rows used", f"{len(df):,}")
    with col2:
        metric_card("Target", TARGET)
    with col3:
        metric_card("Fuel types", df["Fuel"].nunique())

    st.write("Electric, hydrogen, and electricity/hydrogen vehicles are removed because their engine columns are not comparable with gas-side vehicles.")
    st.dataframe(df[DEFAULT_FEATURES + [TARGET]].head(10), use_container_width=True, hide_index=True)

    st.subheader("Select Features")
    all_features = DEFAULT_FEATURES.copy()
    selected_features = st.multiselect(
        "Select explanatory variables",
        options=all_features,
        default=all_features,
    )
    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20)

    if len(selected_features) == 0:
        st.warning("Please select at least one explanatory variable.")
        return

    numeric_features = [col for col in selected_features if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in selected_features if col not in numeric_features]

    X = df[selected_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42,
    )

    model = build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(
        f"Training set: {len(X_train):,} samples &nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Test set: {len(X_test):,} samples",
        unsafe_allow_html=True,
    )

    st.subheader("Model Performance")
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        metric_card("R-squared", f"{r2:.3f}")
    with metric2:
        metric_card("MAE", f"{mae:.1f} g/mi")
    with metric3:
        metric_card("RMSE", f"{rmse:.1f} g/mi")

    st.caption(
        "R-squared measures how much variation in CO2 emissions the model explains. "
        "MAE and RMSE measure prediction error in grams per mile."
    )

    results_df = pd.DataFrame(
        {
            "Actual Comb CO2": y_test,
            "Predicted Comb CO2": y_pred,
        }
    )
    results_df["Error"] = results_df["Predicted Comb CO2"] - results_df["Actual Comb CO2"]

    fig = px.scatter(
        results_df,
        x="Actual Comb CO2",
        y="Predicted Comb CO2",
        color="Error",
        title="Actual vs Predicted Combined CO2",
        labels={
            "Actual Comb CO2": "Actual CO2 (g/mi)",
            "Predicted Comb CO2": "Predicted CO2 (g/mi)",
        },
        opacity=0.7,
    )
    min_value = min(results_df["Actual Comb CO2"].min(), results_df["Predicted Comb CO2"].min())
    max_value = max(results_df["Actual Comb CO2"].max(), results_df["Predicted Comb CO2"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#f97316", width=3, dash="dash"),
            hoverinfo="skip",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Influential Coefficients")
    coef_df = coefficient_table(model, numeric_features, categorical_features)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        "Positive coefficients push predicted CO2 higher; negative coefficients push predicted CO2 lower. "
        "Numeric features are standardized before fitting, so their coefficients can be compared more fairly."
    )

    st.divider()
    st.subheader("Try Your Own Vehicle")
    st.write("Enter vehicle specifications to predict combined CO2 emissions.")

    user_input = {}
    left_col, right_col = st.columns(2)

    for index, feature in enumerate(selected_features):
        target_col = left_col if index % 2 == 0 else right_col
        with target_col:
            if feature in categorical_features:
                options = sorted(df[feature].dropna().unique())
                user_input[feature] = st.selectbox(feature, options)
            else:
                user_input[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    step=1.0 if feature in ["Cyl", "Air Pollution Score", "City MPG", "Hwy MPG", "Cmb MPG"] else 0.1,
                )

    if st.button("Predict CO2 Emissions", use_container_width=True):
        user_df = pd.DataFrame([user_input])[selected_features]
        prediction = model.predict(user_df)[0]
        st.success(f"Predicted combined CO2 emissions: {prediction:.1f} g/mi")

        if prediction < df[TARGET].quantile(0.33):
            st.write("This is lower than most vehicles in the filtered dataset.")
        elif prediction > df[TARGET].quantile(0.67):
            st.write("This is higher than most vehicles in the filtered dataset.")
        else:
            st.write("This is near the middle of the filtered dataset.")
