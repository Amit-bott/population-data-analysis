import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="World Data Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "model.pkl"       # change to your model filename
    scaler_path = "scaler.pkl"     # optional
    imputer_path = "imputer.pkl"   # optional
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
    
    return model, scaler, imputer

model, scaler, imputer = load_model()

# -------------------------------
# LOAD CSV DATA
# -------------------------------
@st.cache_data
def load_csv():
    file_path = "world_population.csv"   # your uploaded CSV
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning("CSV file not found. Upload world_population.csv in repository.")
        return pd.DataFrame()

df = load_csv()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Dashboard Overview", "Data Explorer", "Prediction", "About Notebook"]
)

# -------------------------------
# PAGE 1 – DASHBOARD OVERVIEW
# -------------------------------
if menu == "Dashboard Overview":
    st.title("🌍 World Population Dashboard")
    st.markdown("Professional dashboard for dataset analysis + ML prediction.")

    if not df.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Countries", df["Country"].nunique())

        with col2:
            st.metric("Total Continents", df["Continent"].nunique())

        with col3:
            st.metric("Latest Year", df["Year"].max())

        st.subheader("📈 Population Trend Chart")
        fig = px.line(df, x="Year", y="Population", color="Country", title="Population Over Time")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PAGE 2 – DATA EXPLORER
# -------------------------------
elif menu == "Data Explorer":
    st.title("📊 Data Explorer")

    if df.empty:
        st.error("CSV not loaded.")
    else:
        st.dataframe(df)

        st.subheader("Distribution of Population")
        fig = px.histogram(df, x="Population", nbins=50)
        st.plotly_chart(fig)

        st.subheader("Population by Continent")
        fig2 = px.box(df, x="Continent", y="Population")
        st.plotly_chart(fig2)

# -------------------------------
# PAGE 3 – PREDICTION
# -------------------------------
elif menu == "Prediction":
    st.title("🤖 ML Prediction System")

    if model is None:
        st.error("Model not found. Upload model.pkl to repository.")
    else:
        st.markdown("Enter features below to generate prediction:")

        # Auto-generate numeric inputs
        numeric_cols = ["Feature1", "Feature2", "Feature3"]  # <<< change to your real ML feature names

        inputs = {}
        for c in numeric_cols:
            inputs[c] = st.number_input(c, value=0.0)

        if st.button("Predict"):
            X = pd.DataFrame([inputs])

            if imputer:
                X = imputer.transform(X)
            if scaler:
                X = scaler.transform(X)

            prediction = model.predict(X)[0]

            st.success(f"Prediction: **{prediction}**")

# -------------------------------
# PAGE 4 – ABOUT NOTEBOOK
# -------------------------------
elif menu == "About Notebook":
    st.title("📘 World Population Notebook Summary")

    st.markdown("""
    This section provides information about the analysis performed in the Jupyter notebook.
    
    ### Included Steps
    - Data Cleaning  
    - Exploratory Data Analysis  
    - Visualizations  
    - Feature Engineering  
    - ML Model Training  
    - Evaluation Metrics  
    """)

    st.info("Upload the `.ipynb` file in the repo. It will be shown automatically in future update.")

