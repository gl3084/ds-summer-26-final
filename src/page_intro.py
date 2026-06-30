import streamlit as st
import pandas as pd

def render_descriptive_statistics(df):
    st.markdown('<div class="hero-banner"><h1>📊 Descriptive Statistics</h1></div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

def render_data_quality_check(df):
    st.markdown('<div class="hero-banner"><h1>✅ Data Quality Check</h1></div>', unsafe_allow_html=True)

    # Filter out electric/hydrogen vehicles, which cause the missing values
    df_clean = df[~df["Fuel"].isin(["Electricity", "Hydrogen", "Electricity/Hydrogen"])]

    col_table, col_cards = st.columns([2, 1])

    with col_table:
        missing = df_clean.isnull().sum()
        pct_missing = (missing / len(df_clean) * 100).round(2)
        quality_df = pd.DataFrame({
            "Missing": missing,
            "% Missing": pct_missing
        })
        st.dataframe(quality_df, use_container_width=True, height=380)

    with col_cards:
        completeness = 100 - (df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1]) * 100)
        duplicates = df_clean.duplicated().sum()
        memory_kb = round(df_clean.memory_usage(deep=True).sum() / 1024)

        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Completeness</h3>
            <p>{completeness:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Duplicate Rows</h3>
            <p>{duplicates}</p>
        </div>
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <p>{memory_kb} KB</p>
        </div>
        """, unsafe_allow_html=True)

def render_feature_dictionary():
    st.markdown('<div class="hero-banner"><h1>📖 Feature Dictionary</h1></div>', unsafe_allow_html=True)

    feature_data = [
        ("Model", "Vehicle make and model name", "object"),
        ("Displ", "Engine displacement (liters)", "float64"),
        ("Cyl", "Number of cylinders", "int64"),
        ("Trans", "Transmission type (e.g. Man-6, SCV-7)", "object"),
        ("Drive", "Drivetrain (e.g. 2WD, 4WD)", "object"),
        ("Fuel", "Fuel type (e.g. Gasoline)", "object"),
        ("Cert Region", "Certification region (e.g. CA, FA)", "object"),
        ("Stnd", "Emissions standard code", "object"),
        ("Stnd Description", "Full description of emissions standard", "object"),
        ("Underhood ID", "Engine family identifier code", "object"),
        ("Veh Class", "Vehicle class (e.g. large car)", "object"),
        ("Air Pollution Score", "EPA air pollution score (1–10)", "int64"),
        ("City MPG", "Fuel economy in city driving", "int64"),
        ("Hwy MPG", "Fuel economy in highway driving", "int64"),
        ("Cmb MPG", "Combined city/highway fuel economy", "int64"),
        ("Greenhouse Gas Score", "EPA greenhouse gas score (1–10)", "int64"),
        ("SmartWay", "SmartWay certified (Yes/No)", "object"),
        ("Comb CO2", "Combined CO2 emissions (g/mile)", "int64"),
    ]

    table_rows = "".join(
        f"<tr><td>{feat}</td><td>{desc}</td><td>{dtype}</td></tr>"
        for feat, desc, dtype in feature_data
    )

    st.markdown(f"""
    <div style="overflow-x: auto; max-width: 100%;">
    <table style="width:100%; border-collapse: collapse; border-radius: 10px; overflow: hidden;">
        <thead>
            <tr style="background: #4E7157; color: #E6E8E7; text-align: left;">
                <th style="padding: 12px 16px;">Feature</th>
                <th style="padding: 12px 16px;">Description</th>
                <th style="padding: 12px 16px;">Type</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    </div>
    <style>
        table tr:nth-child(even) {{background-color: #E6E8E7;}}
        table tr:nth-child(odd) {{background-color: #F2F1EC;}}
        table td {{padding: 10px 16px; color: #010C01; border-bottom: 1px solid #ADA78A;}}
        table th {{font-weight: 700;}}
    </style>
    """, unsafe_allow_html=True)

def render():
    st.title("Introduction")
    st.image("assets/car4.gif", use_container_width=True)
    st.write("""Transportation is one of the largest contributors to greenhouse gas emissions worldwide. Our dashboard helps users discover which vehicles produce the lowest carbon emissions and achieve the highest fuel efficiency. Using the Environmental Protection Agency’s official environmental rating system and real-world vehicle specifications, users can compare cars and choose vehicles that would significantly reduce their environmental footprint.
""")
    st.title("Our Dataset")

    # ── Set up dataset ────────────────────────────────────────────
    df = pd.read_excel("CO2.xlsx")

    df_clean = df[~df["Fuel"].isin(["Electricity", "Hydrogen", "Electricity/Hydrogen"])]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", df.shape[1] - 1)  
    with col3:
        st.metric("Target", "CO2 Output")
    with col4:
        st.metric("Source", "Green Vehicle Guide")

    st.write("")  


    tab1, tab2, tab3 = st.tabs(["First rows", "Last rows", "Random sample"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)

    with tab3:
        st.dataframe(df.sample(10), use_container_width=True)

    st.write("")

    render_feature_dictionary()

    st.write("")

    render_data_quality_check(df)

    st.write("")

    render_descriptive_statistics(df_clean)

