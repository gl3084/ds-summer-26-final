import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("CO2.xlsx") 
df_noelectric = df.dropna()

def render():
    st.title("Data Visualization")

    st.write(df_noelectric.describe())

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

    df_noelectric["Manufacturer"] = df_noelectric["Model"].apply(get_manufacturer)

    # Let the user choose a column
    column = st.selectbox("Choose a column", df_noelectric.columns)

    counts = (
        df_noelectric[column]
        .value_counts(dropna=False)
        .reset_index()
    )
    counts.columns = [column, "Count"]

    fig = px.pie(
        counts,
        names=column,
        values="Count",
        title=f"Composition of {column}",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    # ---- THIS WAS MISSING: define make_pie before using it ----
    def make_pie(col_name, key_suffix=""):
        counts = (
            df_noelectric[col_name]
            .value_counts(dropna=False)
            .reset_index()
        )
        counts.columns = [col_name, "Count"]

        fig = px.pie(
            counts,
            names=col_name,
            values="Count",
            title=f"Composition of {col_name}",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, key=f"pie_{key_suffix}")

    # ---- ROW 1: Manufacturer, Vehicle Class ----
    st.title("Data Classifications")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        make_pie("Manufacturer", key_suffix="manufacturer")

    with row1_col2:
        make_pie("Veh Class", key_suffix="vehicle_class")

    # ---- ROW 2: Fuel, Drive, Cylinders ----
    st.subheader("Fuel, Drive & Cylinders")
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        make_pie("Fuel", key_suffix="fuel")

    with row2_col2:
        make_pie("Drive", key_suffix="drive")

    with row2_col3:
        make_pie("Cyl", key_suffix="cyl")


    # ---- BRAND-LEVEL SUMMARY TABLE + BOXPLOT BY VEHICLE CLASS ----
    st.subheader("Fleet Diagnostic: Brand Summary")

    # Force numeric conversion before aggregating (handles stray strings/footnotes)
    df_noelectric["Comb CO2"] = pd.to_numeric(df_noelectric["Comb CO2"], errors="coerce")
    df_noelectric["Cmb MPG"] = pd.to_numeric(df_noelectric["Cmb MPG"], errors="coerce")

    
    brand_summary = (
        df_noelectric.groupby("Manufacturer")
        .agg(
            Avg_Comb_CO2=("Comb CO2", "mean"),
            Avg_Cmb_MPG=("Cmb MPG", "mean"),
            Pct_SmartWay=("SmartWay", lambda x: (x == "Yes").mean() * 100),
            Vehicle_Count=("Model", "count"),
        )
        .reset_index()
        .sort_values("Avg_Comb_CO2", ascending=False)
    )

    brand_summary["Avg_Comb_CO2"] = brand_summary["Avg_Comb_CO2"].round(1)
    brand_summary["Avg_Cmb_MPG"] = brand_summary["Avg_Cmb_MPG"].round(1)
    brand_summary["Pct_SmartWay"] = brand_summary["Pct_SmartWay"].round(1)

    st.dataframe(brand_summary, use_container_width=True)

    st.markdown("**Why do some brands look worse on CO2? Let's check vehicle class.**")

    fig_box = px.box(
        df_noelectric,
        x="Vehicle Class",
        y="Comb CO2",
        title="Comb CO2 Distribution by Vehicle Class",
        points="outliers",
    )
    fig_box.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True, key="box_co2_by_class")

    # ---- IDEA 2: SCATTER - DISPLACEMENT VS COMB CO2, COLORED BY CYLINDERS ----
    st.subheader("Fleet Diagnostic: Displacement vs CO2")

    fig_scatter = px.scatter(
        df_noelectric,
        x="Displ",
        y="Comb CO2",
        color="Cyl",
        trendline="ols",
        title="Engine Displacement vs Comb CO2 (colored by Cylinders)",
        labels={"Displ": "Displacement (L)", "Comb CO2": "Combined CO2 (g/mi)"},
        opacity=0.6,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_displ_co2")

    st.caption(
        "Displacement and cylinder count show a strong relationship with CO2 output — "
        "this is part of why these features are used in the prediction model on the next page."
    )