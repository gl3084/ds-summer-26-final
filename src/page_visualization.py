import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_excel("CO2.xlsx") 
df_noelectric = df.dropna()

def render():
    st.title("This is the visualization page")
    st.write(df_noelectric["Drive"].nunique())

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

        # Check multi-word manufacturers first
        for brand in multi_word_brands:
            if Model.startswith(brand):
                return brand

        # Otherwise return the first word
        return Model.split()[0]

    # Add the new column
    df_noelectric["Manufacturer"] = df_noelectric["Model"].apply(get_manufacturer)

    # Let the user choose a column
    column = st.selectbox("Choose a column", df_noelectric.columns)

    # Count each unique value in the selected column
    counts = (
        df_noelectric[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = [column, "Count"]

    # Interactive pie chart
    fig = px.pie(
        counts,
        names=column,
        values="Count",
        title=f"Composition of {column}",
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig, use_container_width=True)

    
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
        make_pie("Cylinders", key_suffix="cyl")
