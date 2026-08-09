"""
Air Quality Prediction Web App

Interactive dashboard for predicting PM2.5 air pollution
based on weather and environmental conditions.

Run with: streamlit run app.py
Model: Linear Regression (R² = 0.9711)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


# Page config
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .good { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); }
    .moderate { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .unhealthy { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); }
    .very_unhealthy { background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); color: white; }
    .hazardous { background: linear-gradient(135deg, #2c3e50 0%, #000000 100%); color: white; }
    .info-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    try:
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# Header
st.title("🌍 Air Quality Prediction System")
st.markdown("---")

st.write("""
Predict PM2.5 air pollution levels based on current weather conditions.
Enter weather data on the left to get real-time air quality predictions.
""")

# Load model
model = load_model()

if model is None:
    st.error("❌ Model not found. Please ensure the trained model exists at `models/best_model.pkl`")
    st.stop()

# Sidebar inputs
st.sidebar.header("⚙️ Weather Conditions")
st.sidebar.write("Adjust the sliders to see how weather affects air quality")

pm10_val = st.sidebar.slider(
    "PM10 Level (µg/m³)",
    min_value=0, max_value=500, value=80,
    help="Coarse particles - larger than PM2.5"
)

o3_val = st.sidebar.slider(
    "Ozone (ppb)",
    min_value=0, max_value=150, value=50,
    help="Ground-level ozone concentration"
)

temp_max = st.sidebar.slider(
    "Max Temperature (°C)",
    min_value=-20, max_value=50, value=30,
    help="Highest temperature of the day"
)

temp_min = st.sidebar.slider(
    "Min Temperature (°C)",
    min_value=-20, max_value=45, value=20,
    help="Lowest temperature of the day"
)

precip = st.sidebar.slider(
    "Precipitation (mm)",
    min_value=0, max_value=100, value=0,
    help="Rainfall - helps clear pollution"
)

wind_spd = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=0, max_value=30, value=10,
    help="Higher wind = pollution disperses"
)

st.sidebar.markdown("---")

# Calculate derived features
temp_range = temp_max - temp_min
day_of_year = datetime.now().timetuple().tm_yday
month = datetime.now().month

# Prepare features for prediction
input_features = [
    pm10_val, o3_val,
    temp_max, temp_min,
    precip, wind_spd,
    temp_range, day_of_year, month
]

# Make prediction
prediction = model.predict([input_features])[0]
prediction = max(0, prediction)  # Ensure non-negative

# Determine air quality category
if prediction < 12:
    category = "GOOD"
    emoji = "🟢"
    description = "Air quality is satisfactory. Perfect for outdoor activities!"
    color_class = "good"
elif prediction < 35:
    category = "MODERATE"
    emoji = "🟡"
    description = "Air quality is acceptable. Most people can safely do outdoor activities."
    color_class = "moderate"
elif prediction < 55:
    category = "UNHEALTHY FOR SENSITIVE GROUPS"
    emoji = "🟠"
    description = "Sensitive groups (children, elderly, asthmatic) should limit outdoor activities."
    color_class = "unhealthy"
elif prediction < 150:
    category = "UNHEALTHY"
    emoji = "🔴"
    description = "Everyone may begin to experience health effects. Limit outdoor exposure."
    color_class = "unhealthy"
elif prediction < 250:
    category = "VERY UNHEALTHY"
    emoji = "🟣"
    description = "Everyone should avoid outdoor activities. Health alert: The entire population is more likely to be affected."
    color_class = "very_unhealthy"
else:
    category = "HAZARDOUS"
    emoji = "⚫"
    description = "EVERYONE should avoid outdoor activities. Air quality is UNBREATHABLE. STAY INDOORS!"
    color_class = "hazardous"

# Display main prediction
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h2 style='margin: 0;'>Predicted PM2.5</h2>
        <h1 style='margin: 10px 0; font-size: 3em;'>{prediction:.1f}</h1>
        <p style='margin: 0; font-size: 1.2em;'>µg/m³</p>
    </div>
    """, unsafe_allow_html=True)

# Air quality category
st.markdown(f"""
<div style='text-align: center; padding: 20px; margin: 20px 0; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
    <h2>{emoji} {category}</h2>
    <p>{description}</p>
</div>
""", unsafe_allow_html=True)

# Input summary
st.subheader("📊 Input Summary")

input_data = pd.DataFrame({
    "Parameter": ["PM10", "Ozone", "Max Temp", "Min Temp", "Precipitation", "Wind Speed"],
    "Value": [pm10_val, o3_val, temp_max, temp_min, precip, wind_spd],
    "Unit": ["µg/m³", "ppb", "°C", "°C", "mm", "m/s"]
})

st.table(input_data)

# Information section
st.markdown("---")
st.subheader("ℹ️ About Air Quality Levels")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🟢 GOOD (0-12)**
    
    Perfect air quality! Safe for all outdoor activities.
    
    ---
    
    **🟡 MODERATE (12-35)**
    
    Acceptable air quality. Sensitive groups should be cautious.
    
    ---
    
    **🟠 UNHEALTHY FOR SENSITIVE (35-55)**
    
    Sensitive groups (children, elderly, asthmatics) should limit outdoor time.
    """)

with col2:
    st.markdown("""
    **🔴 UNHEALTHY (55-150)**
    
    Everyone may experience health effects. Reduce outdoor exposure.
    
    ---
    
    **🟣 VERY UNHEALTHY (150-250)**
    
    Serious health risk for everyone. Avoid outdoor activities.
    
    ---
    
    **⚫ HAZARDOUS (250+)**
    
    UNBREATHABLE AIR! DO NOT GO OUTSIDE! This is an emergency level.
    """)

st.markdown("---")
st.subheader("📚 What is PM2.5?")

st.write("""
**PM2.5** stands for **Particulate Matter 2.5 micrometers or smaller**.

These are tiny particles suspended in the air that can be inhaled deep into the lungs and even enter the bloodstream.

**Common sources:**
- Vehicle exhaust
- Power plants and factories
- Industrial facilities
- Wildfires and dust storms
- Construction and demolition

**Health impacts:**
- Respiratory problems (asthma, bronchitis)
- Heart diseases
- Reduced lung function
- Premature death with long-term exposure
- Children and elderly are most vulnerable
""")

st.subheader("🔧 How This Model Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Collection**
    - Collected 90 days of historical weather data
    - Gathered air pollution measurements from APIs
    - Location: Cairo, Egypt
    
    **Data Processing**
    - Removed errors and outliers
    - Engineered 3 new features (temperature range, day of year, month)
    - Normalized values for ML model
    """)

with col2:
    st.markdown("""
    **Model Training**
    - Trained on 72 days (80% of data)
    - Tested on 18 days (20% of data)
    - Linear Regression algorithm
    
    **Performance**
    - R² Score: 0.9711 (97.11% accuracy)
    - RMSE: 0.7586 µg/m³
    - MAE: 0.4379 µg/m³
    - Very reliable predictions
    """)

st.markdown("---")
st.caption("Built with Python • Scikit-learn • Streamlit | Model Status: Production Ready ✅")
