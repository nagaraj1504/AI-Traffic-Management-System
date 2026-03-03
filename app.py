import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(page_title="AI Traffic Management System", layout="wide")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Train.csv")
    return df

df = load_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

df = df.select_dtypes(include=np.number)

# =========================================================
# TRAIN MODELS
# =========================================================
@st.cache_resource
def train_models(df):

    X = df.drop("traffic_volume", axis=1)
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=42
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    df['congestion'] = df['traffic_volume'].apply(
        lambda x: 0 if x < 2000 else 1 if x < 4000 else 2
    )

    Xc = df.drop(['traffic_volume', 'congestion'], axis=1)
    yc = df['congestion']

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        random_state=42
    )
    clf.fit(Xc_train, yc_train)

    yc_pred = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, yc_pred)

    return reg, clf, r2, acc

traffic_model, congestion_model, r2_value, acc_value = train_models(df)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Traffic Prediction",
        "Congestion Level",
        "Emission Estimation",
        "Signal Optimization",
        "Traffic Heatmap",
        "Model Performance"
    ]
)

st.sidebar.header("Input Conditions")

temperature = st.sidebar.slider("Temperature", -20, 50, 20)
humidity = st.sidebar.slider("Humidity", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed", 0, 50, 10)
hour = st.sidebar.slider("Hour", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 6)
is_weekend = st.sidebar.selectbox("Weekend?", [0, 1])

feature_vector = np.array([[temperature, humidity, wind_speed,
                            hour, day_of_week, month, is_weekend]])

st.title("🚦 AI-Based Intelligent Traffic Management System")

# =========================================================
# MODULES
# =========================================================

if page == "Traffic Prediction":

    prediction = traffic_model.predict(feature_vector)[0]
    st.metric("Predicted Traffic Volume", int(prediction))


elif page == "Congestion Level":

    congestion_pred = congestion_model.predict(feature_vector)[0]

    if congestion_pred == 0:
        st.success("Low Congestion")
    elif congestion_pred == 1:
        st.warning("Medium Congestion")
    else:
        st.error("High Congestion")


elif page == "Emission Estimation":

    prediction = traffic_model.predict(feature_vector)[0]
    emission_factor = 0.00027
    emission = prediction * emission_factor

    st.metric("Estimated CO₂ Emission (tons/hour)", round(emission, 3))


elif page == "Signal Optimization":

    prediction = traffic_model.predict(feature_vector)[0]

    if prediction < 2000:
        green_time = 30
    elif prediction < 4000:
        green_time = 60
    else:
        green_time = 90

    red_time = 120 - green_time

    col1, col2 = st.columns(2)
    col1.metric("Green Signal Time (sec)", green_time)
    col2.metric("Red Signal Time (sec)", red_time)


elif page == "Traffic Heatmap":

    fig = px.density_heatmap(
        df,
        x="hour",
        y="day_of_week",
        z="traffic_volume",
        nbinsx=24,
        nbinsy=7
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Model Performance":

    col1, col2 = st.columns(2)
    col1.metric("Regression R² Score", round(r2_value, 3))
    col2.metric("Classification Accuracy", round(acc_value, 3))