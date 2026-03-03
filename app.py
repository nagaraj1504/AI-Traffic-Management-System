import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Traffic Management System", layout="wide")

# Load models
traffic_model = pickle.load(open("models/best_model.pkl", "rb"))
congestion_model = pickle.load(open("models/congestion_model.pkl", "rb"))

# Load dataset for heatmap & comparison
df = pd.read_csv("data/final_dataset.csv")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Traffic Prediction",
        "Congestion Level",
        "Emission Estimation",
        "Signal Optimization",
        "Traffic Heatmap",
        "Model Comparison"
    ]
)

# Sidebar Input Controls
st.sidebar.header("Input Conditions")

temperature = st.sidebar.slider("Temperature", -20, 50, 20)
humidity = st.sidebar.slider("Humidity", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed", 0, 50, 10)
hour = st.sidebar.slider("Hour", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 6)
is_weekend = st.sidebar.selectbox("Weekend?", [0, 1])
rain = st.sidebar.slider("Rain (mm)", 0.0, 10.0, 0.0)
snow = st.sidebar.slider("Snow (mm)", 0.0, 10.0, 0.0)
air_pollution = st.sidebar.slider("Air Pollution Index", 0, 500, 50)
visibility = st.sidebar.slider("Visibility", 0, 20, 10)
clouds = st.sidebar.slider("Cloud Coverage", 0, 100, 50)

# Feature vector (MUST match training order)
features = np.array([[0, air_pollution, humidity, wind_speed, 0,
                      visibility, 0, temperature, rain, snow,
                      clouds, 0, 0,
                      hour, day_of_week, month, is_weekend]])

prediction = traffic_model.predict(features)[0]
congestion_pred = congestion_model.predict(features)[0]

# ================================
# MODULES
# ================================

if page == "Traffic Prediction":

    st.title("🚦 Traffic Volume Prediction")
    st.metric("Predicted Traffic Volume", int(prediction))


elif page == "Congestion Level":

    st.title("🚗 Congestion Classification")

    if congestion_pred == 0:
        st.success("Low Congestion")
    elif congestion_pred == 1:
        st.warning("Medium Congestion")
    else:
        st.error("High Congestion")


elif page == "Emission Estimation":

    st.title("🌍 CO₂ Emission Estimation")

    emission_factor = 0.00027
    emission = prediction * emission_factor

    st.metric("Estimated CO₂ Emission (tons/hour)", round(emission, 3))


elif page == "Signal Optimization":

    st.title("🚥 Smart Signal Timing Optimization")

    if prediction < 2000:
        green_time = 30
    elif prediction < 4000:
        green_time = 60
    else:
        green_time = 90

    red_time = 120 - green_time

    st.write(f"Green Time: {green_time} seconds")
    st.write(f"Red Time: {red_time} seconds")


elif page == "Traffic Heatmap":

    st.title("🔥 Traffic Heatmap")

    fig = px.density_heatmap(
        df,
        x="hour",
        y="day_of_week",
        z="traffic_volume",
        nbinsx=24,
        nbinsy=7
    )

    st.plotly_chart(fig)


elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    scores = pd.read_csv("models/model_scores.csv")

    fig = px.bar(scores, x="Model", y="R2", text="R2")

    st.plotly_chart(fig)
