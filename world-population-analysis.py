import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="World Population Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    imputer = joblib.load("imputer.pkl") if os.path.exists("imputer.pkl") else None
    return model, scaler, imputer

model, scaler, imputer = load_model()

# ---------------- LOAD CSV ----------------
@st.cache_data
def load_data():
    return pd.read_csv("world_population.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Dashboard Overview", "Data Explorer", "Prediction", "About"]
)


if menu == "Dashboard Overview":
    st.title("🌍 World Population Dashboard")
    st.markdown("Professional interactive dashboard based on real world population data.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌐 Total Countries", df["Country/Territory"].nunique())

    with col2:
        st.metric("🗺 Total Continents", df["Continent"].nunique())

    with col3:
        st.metric("📅 Latest Population Year", "2022")

    # ---- Convert wide data to long format ----
    pop_cols = [
        '1970 Population', '1980 Population', '1990 Population',
        '2000 Population', '2010 Population',
        '2015 Population', '2020 Population', '2022 Population'
    ]

    df_long = df.melt(
        id_vars=["Country/Territory", "Continent"],
        value_vars=pop_cols,
        var_name="Year",
        value_name="Population"
    )

    df_long["Year"] = df_long["Year"].str.extract(r'(\d+)').astype(int)

    st.subheader("📈 Population Growth Over Time")
    fig = px.line(
        df_long,
        x="Year",
        y="Population",
        color="Country/Territory",
        title="Population Growth by Country"
    )
    st.plotly_chart(fig, use_container_width=True)


elif menu == "Data Explorer":
    st.title("📊 Data Explorer")

    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Population Distribution (2022)")
    fig1 = px.histogram(df, x="2022 Population", nbins=40)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Population by Continent (2022)")
    fig2 = px.box(df, x="Continent", y="2022 Population")
    st.plotly_chart(fig2, use_container_width=True)


elif menu == "Prediction":
    st.title("🤖 Population Prediction")

    if model is None:
        st.warning("⚠ model.pkl not found. Upload trained ML model.")
    else:
        st.markdown("Predict future population using past data.")

        pop_2015 = st.number_input("2015 Population", min_value=0)
        pop_2020 = st.number_input("2020 Population", min_value=0)
        growth_rate = st.number_input("Growth Rate", value=0.0)

        if st.button("Predict"):
            X = pd.DataFrame([[pop_2015, pop_2020, growth_rate]],
                             columns=["2015 Population", "2020 Population", "Growth Rate"])

            if imputer:
                X = imputer.transform(X)
            if scaler:
                X = scaler.transform(X)

            prediction = model.predict(X)[0]
            st.success(f"✅ Predicted Population: {int(prediction):,}")


else:
    st.title("📘 About Project")
    st.markdown("""
    ### 🌍 World Population Analysis Dashboard

    **Features**
    - Interactive charts
    - Country & continent level analysis
    - Time-series population growth
    - Machine Learning based prediction
    - Streamlit web application

    **Dataset Source**
    - World Population Dataset (1970–2022)
    """)

