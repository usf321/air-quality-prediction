"""
Air Quality Prediction System

Professional ML-powered dashboard for real-time PM2.5 forecasting.
Delivers accurate air pollution predictions based on weather conditions.

Deployment: Streamlit Cloud
Model: Linear Regression (R² = 0.97)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Air Quality Prediction | Real-Time PM2.5 Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: #f8f9fa;
    }
    
    /* Header */
    .header-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    /* Prediction card - main metric */
    .prediction-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        margin: 20px 0;
    }
    
    .prediction-card h2 {
        margin: 0;
        font-size: 18px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-card h1 {
        margin: 15px 0;
        font-size: 48px;
        font-weight: bold;
    }
    
    .prediction-card p {
        margin: 0;
        font-size: 16px;
    }
    
    /* Category badges */
    .category-badge {
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .badge-good { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .badge-moderate { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .badge-sensitive { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
    .badge-unhealthy { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .badge-very-unhealthy { background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%); }
    .badge-hazardous { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); }
    
    .category-badge h2 {
        margin: 0 0 10px 0;
        font-size: 24px;
    }
    
    .category-badge p {
        margin: 0;
        font-size: 16px;
        opacity: 0.95;
    }
    
    /* Info boxes */
    .info-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .info-section h3 {
        color: #1e293b;
        margin-top: 0;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-box h4 {
        color: #64748b;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 10px 0;
    }
    
    .metric-box .value {
        font-size: 28px;
        font-weight: bold;
        color: #1e293b;
        margin: 0;
    }
    
    .metric-box .unit {
        color: #94a3b8;
        font-size: 12px;
        margin: 5px 0 0 0;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 30px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained Linear Regression model from disk."""
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError:
        return None


# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown("""
<div class='header-section'>
    <h1 style='margin: 0; font-size: 32px;'>🌍 Air Quality Prediction System</h1>
    <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;'>
        Real-time PM2.5 forecasting powered by machine learning
    </p>
</div>
""", unsafe_allow_html=True)

st.write("""
Accurate air quality predictions based on weather conditions. 
Adjust weather parameters below to see how environmental factors influence air pollution levels.
""")

# Load model
model = load_model()

if model is None:
    st.error("❌ Model not found. Please ensure the trained model exists at `models/best_model.pkl`")
    st.stop()

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.header("⚙️ Weather Parameters")
st.sidebar.write("Set current weather conditions to predict air quality")

pm10_val = st.sidebar.slider(
    "PM10 Level (µg/m³)",
    min_value=0, max_value=500, value=80,
    help="Coarse particulate matter - larger particles suspended in air"
)

o3_val = st.sidebar.slider(
    "Ozone (ppb)",
    min_value=0, max_value=150, value=50,
    help="Ground-level ozone concentration - formed from vehicle and industrial emissions"
)

temp_max = st.sidebar.slider(
    "Max Temperature (°C)",
    min_value=-20, max_value=50, value=30,
    help="Daily maximum temperature"
)

temp_min = st.sidebar.slider(
    "Min Temperature (°C)",
    min_value=-20, max_value=45, value=20,
    help="Daily minimum temperature"
)

precip = st.sidebar.slider(
    "Precipitation (mm)",
    min_value=0, max_value=100, value=0,
    help="Rainfall - higher precipitation clears pollutants from the air"
)

wind_spd = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=0, max_value=30, value=10,
    help="Higher wind speeds help disperse pollution"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Increase wind speed and precipitation to see how they reduce pollution levels")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

temp_range = temp_max - temp_min
day_of_year = datetime.now().timetuple().tm_yday
month = datetime.now().month

# Prepare input features in correct order
input_features = [
    pm10_val, o3_val,
    temp_max, temp_min,
    precip, wind_spd,
    temp_range, day_of_year, month
]

# ============================================================================
# PREDICTION
# ============================================================================

prediction = model.predict([input_features])[0]
prediction = max(0, prediction)  # Ensure non-negative

# Determine air quality category
if prediction < 12:
    category = "GOOD"
    emoji = "🟢"
    description = "Air quality is satisfactory. Excellent for all outdoor activities!"
    color_class = "badge-good"
elif prediction < 35:
    category = "MODERATE"
    emoji = "🟡"
    description = "Air quality is acceptable. Most people can safely participate in outdoor activities."
    color_class = "badge-moderate"
elif prediction < 55:
    category = "UNHEALTHY FOR SENSITIVE GROUPS"
    emoji = "🟠"
    description = "Children, elderly, and people with respiratory conditions should limit outdoor exposure."
    color_class = "badge-sensitive"
elif prediction < 150:
    category = "UNHEALTHY"
    emoji = "🔴"
    description = "General population may experience health effects. Reduce prolonged outdoor activities."
    color_class = "badge-unhealthy"
elif prediction < 250:
    category = "VERY UNHEALTHY"
    emoji = "🟣"
    description = "Everyone should avoid outdoor activities. Serious health risk for the entire population."
    color_class = "badge-very-unhealthy"
else:
    category = "HAZARDOUS"
    emoji = "⚫"
    description = "Air quality is UNBREATHABLE. DO NOT GO OUTSIDE. This is a health emergency."
    color_class = "badge-hazardous"

# ============================================================================
# MAIN PREDICTION DISPLAY
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div class='prediction-card'>
        <h2>Predicted PM2.5</h2>
        <h1>{prediction:.1f}</h1>
        <p>µg/m³</p>
    </div>
    """, unsafe_allow_html=True)

# Air quality category badge
st.markdown(f"""
<div class='category-badge {color_class}'>
    <h2>{emoji} {category}</h2>
    <p>{description}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INPUT SUMMARY TABLE
# ============================================================================

st.subheader("📊 Current Conditions Summary")

input_df = pd.DataFrame({
    "Parameter": ["PM10", "Ozone", "Max Temp", "Min Temp", "Precipitation", "Wind Speed"],
    "Value": [pm10_val, o3_val, temp_max, temp_min, precip, wind_spd],
    "Unit": ["µg/m³", "ppb", "°C", "°C", "mm", "m/s"]
})

st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================================
# AIR QUALITY INFORMATION
# ============================================================================

st.markdown("---")
st.subheader("📚 Air Quality Classification Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='info-section'>
        <h3>🟢 GOOD (0-12)</h3>
        <p><strong>Air Quality Index:</strong> 0-50</p>
        <p>Perfect air quality. Safe for all groups including children and elderly. Ideal for outdoor activities.</p>
    </div>
    
    <div class='info-section'>
        <h3>🟡 MODERATE (12-35)</h3>
        <p><strong>Air Quality Index:</strong> 51-100</p>
        <p>Acceptable air quality. Most people can engage in outdoor activities safely.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='info-section'>
        <h3>🟠 UNHEALTHY FOR SENSITIVE (35-55)</h3>
        <p><strong>Air Quality Index:</strong> 101-150</p>
        <p>Members of sensitive groups (children, elderly, asthmatics) should limit outdoor exposure.</p>
    </div>
    
    <div class='info-section'>
        <h3>🔴 UNHEALTHY (55-150)</h3>
        <p><strong>Air Quality Index:</strong> 151-200</p>
        <p>General population begins to experience health effects. Limit outdoor activities.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='info-section'>
        <h3>🟣 VERY UNHEALTHY (150-250)</h3>
        <p><strong>Air Quality Index:</strong> 201-300</p>
        <p>Everyone should avoid outdoor activities. Serious health risk.</p>
    </div>
    
    <div class='info-section'>
        <h3>⚫ HAZARDOUS (250+)</h3>
        <p><strong>Air Quality Index:</strong> 301+</p>
        <p><strong>HEALTH EMERGENCY.</strong> Everyone should remain indoors.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DETAILED INFORMATION
# ============================================================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📖 About PM2.5", "🔧 Model Details", "💡 Tips"])

with tab1:
    st.subheader("What is PM2.5?")
    
    st.write("""
    **PM2.5** refers to **Particulate Matter with a diameter of 2.5 micrometers or less** - 
    about 30 times smaller than a human hair.
    
    These tiny particles can remain suspended in the air for days and travel long distances. 
    When inhaled, they penetrate deep into the lungs and can even enter the bloodstream.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Common Sources:**
        - Vehicle exhaust
        - Power plants and factories
        - Industrial facilities
        - Wildfires and dust storms
        - Construction activities
        - Cooking emissions
        """)
    
    with col2:
        st.markdown("""
        **Health Impacts:**
        - Respiratory problems (asthma, bronchitis)
        - Cardiovascular diseases
        - Reduced lung function
        - Decreased life expectancy
        - Vulnerable groups: children, elderly, people with respiratory/heart conditions
        """)

with tab2:
    st.subheader("Model Architecture & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:** Linear Regression
        
        **Training Data:**
        - Time Period: 90+ days
        - Location: Cairo, Egypt
        - Samples: 72 days (training)
        - Validation: 18 days (testing)
        
        **Input Features:**
        1. PM10 (µg/m³)
        2. Ozone (ppb)
        3. Max Temperature (°C)
        4. Min Temperature (°C)
        5. Precipitation (mm)
        6. Wind Speed (m/s)
        7. Temperature Range (engineered)
        8. Day of Year (engineered)
        9. Month (engineered)
        """)
    
    with col2:
        st.markdown("""
        **Performance Metrics:**
        
        | Metric | Value |
        |--------|-------|
        | R² Score | 0.9711 |
        | RMSE | 0.7586 µg/m³ |
        | MAE | 0.4379 µg/m³ |
        | Accuracy | 97.1% |
        
        **Why Linear Regression?**
        - Better generalization on small dataset
        - Fewer parameters (less overfitting)
        - Interpretable results
        - Faster predictions
        """)

with tab3:
    st.subheader("How to Improve Air Quality")
    
    st.markdown("""
    **For Individuals:**
    - Monitor air quality alerts regularly
    - Use N95/N99 masks during high pollution days
    - Stay indoors when air quality is hazardous
    - Exercise indoors on poor air quality days
    - Use air purifiers in your home
    
    **For Communities:**
    - Support renewable energy initiatives
    - Advocate for stricter emission standards
    - Promote public transportation usage
    - Plant trees and greenery
    - Support air quality monitoring programs
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔗 GitHub: [usf321/air-quality-prediction](https://github.com/usf321/air-quality-prediction)")

with col2:
    st.caption("📊 Model Status: Production Ready")

with col3:
    st.caption("✅ Last Updated: August 2026")

st.caption("---")
st.caption("Built with Python • Scikit-learn • Streamlit | Air Quality Prediction System v1.1")