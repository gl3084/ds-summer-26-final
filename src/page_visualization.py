import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("CO2.xlsx") 
df_noelectric = df.dropna()

def render():
    st.title("Data Visualization")

    # ── Numeric variable stats explorer ────────────────────────────────────
    st.markdown("### 📊 Variable Statistics Explorer")

    # CSS for metric cards (insert here, before the cards are rendered)
    st.markdown("""
    <style>
    .metric-card {
        background: #37ddfa;
        border-left: 5px solid #4bc920;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 13px;
        color: white;
        margin-bottom: 4px;
    }
    .metric-card p {
        font-size: 20px;
        font-weight: bold;
        color: white;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    numeric_vars = df_noelectric.select_dtypes(include="number").columns.tolist()
    selected_var = st.selectbox("Choose a numerical variable", numeric_vars)

    stats = df_noelectric[selected_var].describe()

    stat_labels = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    cols = st.columns(len(stat_labels))

    for col, stat_name in zip(cols, stat_labels):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stat_name.upper()}</h3>
                <p>{stats[stat_name]:.2f}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")







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






    





    st.subheader("Distribution of Greenhouse Gas Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df_noelectric,
        x="Greenhouse Gas Score",
        bins=10,
        kde=True,
        ax=ax,
        color="steelblue",
    )
    ax.set_title("Distribution of Greenhouse Gas Score")
    ax.set_xlabel("Greenhouse Gas Score")
    ax.set_ylabel("Count")

    st.pyplot(fig)




    # CONVERTING / INTO REGULAR MPG AVERAGE.
    def parse_mpg(value):
        value = str(value)
        if "/" in value:
            parts = value.split("/")
            try:
                nums = [float(p) for p in parts]
                return sum(nums) / len(nums)
            except ValueError:
                return None
        try:
            return float(value)
        except ValueError:
            return None

    # Apply to any MPG column that might have this issue
    for col in ["City MPG", "Hwy MPG", "Cmb MPG"]:
        df_noelectric[col] = df_noelectric[col].apply(parse_mpg)


    st.subheader("Feature Correlation Heatmap")

    numeric_cols = ["Displ", "Cyl", "City MPG", "Hwy MPG", "Cmb MPG",
                    "Air Pollution Score", "Greenhouse Gas Score", "Comb CO2"]

    corr = df_noelectric[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Numeric Features")

    st.pyplot(fig)


    # ── Top correlations with target (two columns) ────────────────────────────────
    def show_top_correlations(target_col, corr_matrix):
        target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        st.markdown(f"**Top features correlated with `{target_col}`:**")
        for feat, val in target_corr.head(5).items():
            direction = "+" if corr_matrix.loc[feat, target_col] > 0 else "−"
            bar_width = int(val * 100)
            st.markdown(
                f"- **{feat}** → {direction}{val:.3f} "
                f'<span style="display:inline-block;height:10px;width:{bar_width}px;'
                f'background:#57068C;border-radius:4px;"></span>',
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)

    with col1:
        show_top_correlations("Comb CO2", corr)

    with col2:
        show_top_correlations("Greenhouse Gas Score", corr)

    st.markdown("---")




    # ── Target + feature setup ────────────────────────────────────
    target = st.selectbox("Choose target variable", ["Comb CO2", "Greenhouse Gas Score"])

    features = [
        "Displ", "Cyl", "City MPG", "Hwy MPG", "Cmb MPG",
        "Air Pollution Score", "Greenhouse Gas Score", "Comb CO2"
    ]
    # Remove the target itself from the feature list, in case it's in there
    features = [f for f in features if f != target]


    
    # ── 4. Scatter plot explorer ────────────────────────────────────
    st.markdown("### 🔗 Feature vs Target Explorer")
    col_a, col_b = st.columns(2)
    with col_a:
        x_feat = st.selectbox("X-axis feature", features, index=0)
    with col_b:
        color_feat = st.selectbox(
            "Color by (optional)", ["None"] + features, index=0
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_feat == "None":
        sns.scatterplot(
            data=df_noelectric,
            x=x_feat,
            y=target,
            alpha=0.5,
            color="purple",
            ax=ax,
        )
    elif pd.api.types.is_numeric_dtype(df_noelectric[color_feat]):
        # Continuous color variable — use a colorbar scatter
        scatter = ax.scatter(
            df_noelectric[x_feat],
            df_noelectric[target],
            c=df_noelectric[color_feat],
            cmap="Purples",
            alpha=0.5,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_feat)
    else:
        # Categorical color variable — use hue
        sns.scatterplot(
            data=df_noelectric,
            x=x_feat,
            y=target,
            hue=color_feat,
            palette="Purples",
            alpha=0.6,
            ax=ax,
        )
        ax.legend(title=color_feat, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Trendline (no statsmodels needed)
    sns.regplot(
        data=df_noelectric,
        x=x_feat,
        y=target,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2, "linestyle": "--"},
        ax=ax,
    )

    ax.set_title(f"{x_feat} vs {target}")
    ax.set_xlabel(x_feat)
    ax.set_ylabel(target)

    st.pyplot(fig)
    st.markdown("---")