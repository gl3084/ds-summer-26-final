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




    # ---- SHARED COLOR MAPPING FOR VEHICLE CLASS (used by both charts) ----
    veh_classes = sorted(df_noelectric["Veh Class"].unique())
    palette_colors = sns.color_palette("tab20", n_colors=len(veh_classes))
    class_color_map = dict(zip(veh_classes, palette_colors))







    # Let the user choose a column.      GET RID OF THIS THING!!! i don't need a customizable pie chart selector.
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

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x="Veh Class",
        y="Comb CO2",
        data=df_noelectric,
        hue="Veh Class",
        palette=class_color_map,
        ax=ax,
        legend=False,
    )
    ax.set_title("Comb CO2 Distribution by Vehicle Class")
    ax.set_xlabel("Vehicle Class")
    ax.set_ylabel("Combined CO2 (g/mi)")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)









    # Only keep brands with at least N vehicles in the dataset (filters out 1-2 car exotic brands)
    min_vehicles = 5
    brand_counts = df_noelectric["Manufacturer"].value_counts()
    valid_brands = brand_counts[brand_counts >= min_vehicles].index

    df_filtered = df_noelectric[df_noelectric["Manufacturer"].isin(valid_brands)]

    composition = (
        pd.crosstab(df_filtered["Manufacturer"], df_filtered["Veh Class"], normalize="index")
        * 100
    )

    co2_order = (
        df_filtered.groupby("Manufacturer")["Comb CO2"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    composition = composition.loc[co2_order]

















    st.markdown("**Brand composition by vehicle class** — see which brands lean toward higher-emission classes")

    # Build a % composition table: rows = Manufacturer, cols = Veh Class
    composition = (
        pd.crosstab(df_noelectric["Manufacturer"], df_noelectric["Veh Class"], normalize="index")
        * 100
    )

    co2_order = (
        df_noelectric.groupby("Manufacturer")["Comb CO2"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    composition = composition.loc[co2_order]

    # Build the color list in the SAME column order as `composition`
    bar_colors = [class_color_map[col] for col in composition.columns]

    fig, ax = plt.subplots(figsize=(10, 12))
    composition.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=bar_colors,
        width=0.8,
    )
    ax.set_title("Vehicle Class Composition by Brand (sorted by avg Comb CO2, high to low)")
    ax.set_xlabel("% of Brand's Lineup")
    ax.set_ylabel("Manufacturer")
    ax.legend(title="Vehicle Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.invert_yaxis()

    st.pyplot(fig)












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