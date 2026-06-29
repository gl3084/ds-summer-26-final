import streamlit as st
import pandas as pd

def render():
    st.title("Intro Page")
    st.image("assets/car.jpg", use_container_width=True)
    st.write("This should show up")

    # ── Set up dataset ────────────────────────────────────────────
    df = pd.read_excel("CO2.xlsx")

    st.title("THIS IS ALL TESTING SCROLL DOWN TO SEE YOUR PAGE")
    st.dataframe(df.head(5))

    st.write(df.shape)
    st.write(df["Fuel"].value_counts())

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")
            
    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Original Dataset Stats"):
        st.dataframe(df.describe())


    # Removed electric cars which are all the ones with missing values
    st.markdown("##### NO ELECTRIC OR HYBRID CAR DATASET")
    df_noelectric = df[df["Fuel"] == "Gasoline"]
    st.dataframe(df_noelectric.sample(30))
    st.write(df_noelectric.dtypes)
    st.write(df_noelectric.columns)
    st.write(df_noelectric["Greenhouse Gas Score"].value_counts().sort_index())
    st.write(df_noelectric.shape)

    st.markdown("##### Missing values of Only Gas Cars")
    missing = df_noelectric.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")
            
    st.markdown("##### 📈 Summary Statistics of only Gas Cars")
    if st.button("Show Updated Dataset Stats"):
        st.dataframe(df_noelectric.describe())

    st.markdown("##### GAS AND HYBRID CAR DATASET")
    df_gas_and_hybird = df[df["Fuel"] != "Electricity"]
    df_gas_and_hybird = df_gas_and_hybird[df_gas_and_hybird["Fuel"] != "Hydrogen"]
    df_gas_and_hybird = df_gas_and_hybird[df_gas_and_hybird["Fuel"] != "Electricity/Hydrogen"]
    st.write(df_gas_and_hybird["Fuel"].value_counts())

    st.divider()
