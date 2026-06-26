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
