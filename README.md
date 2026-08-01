🌍 Air Quality Prediction System

End-to-End ML Pipeline for PM2.5 Air Pollution Forecasting

📋 Table of Contents
Goals
Overview
Dataset
Model Performance
System Architecture
Quick Start
Project Structure
Technologies
What I Learned
Future Research
Data Sources
🎯 Goals

This project demonstrates my ability to build complete machine learning systems. Specific objectives:

✅ Learn ML Pipelines - Data collection → cleaning → training → deployment
✅ Practice Feature Engineering - Create meaningful features from raw data
✅ Deploy Real Application - Build an interactive web app
✅ Predict PM2.5 - Forecast air pollution from weather conditions
✅ Build Portfolio Project - Showcase skills for scholarship/job applications

📌 Overview

This is an end-to-end machine learning project that predicts PM2.5 air pollution based on weather conditions.

What it does:

Collects real historical weather and air quality data from APIs
Cleans and engineers features from raw data
Trains and compares multiple ML models
Deploys an interactive Streamlit web application
Makes real-time air quality predictions

Who this is for:

Portfolio building for scholarships
Demonstrating ML fundamentals
Learning complete ML workflows
Understanding data science best practices
📊 Dataset
Metric	Details
Time Period	90 days (April - July 2024)
Location	Cairo, Egypt (30.04°N, 31.24°E)
Total Samples	90 observations
Features	9 (6 original + 3 engineered)
Target Variable	PM2.5 (µg/m³)
Feature Details
Feature	Type	Source	Purpose
PM10	Pollutant	API	Coarse particles
Ozone	Pollutant	API	Ground-level ozone
Max Temperature	Weather	API	Daily high temp
Min Temperature	Weather	API	Daily low temp
Precipitation	Weather	API	Rainfall effect
Wind Speed	Weather	API	Pollution dispersion
Temp Range	Engineered	Max - Min	Daily volatility
Day of Year	Engineered	1-365	Seasonal pattern
Month	Engineered	1-12	Monthly trend
📈 Model Performance
Model Comparison

Two models were trained on 72 days and evaluated on 18 days:

Linear Regression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R² Score:    0.9711
RMSE:        0.7586 µg/m³
MAE:         0.4379 µg/m³
Training:    Fast, interpretable

Random Forest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R² Score:    0.9656
RMSE:        0.8275 µg/m³
MAE:         0.6315 µg/m³
Training:    Moderate complexity

🏆 WINNER: Linear Regression
Performance Metrics Explained
Metric	Formula	Interpretation
R² Score	1 - (SS_res / SS_tot)	% of variance explained (0.97 = 97%)
RMSE	√(Σ(y_pred - y_true)²/n)	Average prediction error (~0.76 µg/m³)
MAE	Σ(|y_pred - y_true|)/n	Mean absolute deviation (~0.44 µg/m³)
Why Linear Regression Won

Although Random Forest is theoretically more powerful at learning complex relationships, the dataset is relatively small (90 observations). Under these conditions:

Generalization - Linear Regression avoided overfitting
Parsimony - Simpler model, fewer parameters to tune
Interpretability - Coefficients show feature importance clearly
Stability - Less sensitive to noise in small datasets

This demonstrates Occam's Razor: the simpler explanation is often better.

🏗️ System Architecture
Data Pipeline
Open-Meteo APIs
    │
    ├─ Weather Archive
    │  (Temperature, Precipitation, Wind)
    │
    └─ Air Quality API
       (PM2.5, PM10, Ozone)
    │
    ▼
Data Collection Module
    │
    ▼
Raw Data Storage
    (data/raw/air_quality_data.csv)
    │
    ▼
Preprocessing Module
    │
    ├─ Handle missing values
    ├─ Remove outliers (Z-score > 3)
    └─ Create derived features
    │
    ▼
Processed Data Storage
    (data/processed/air_quality_clean.csv)
    │
    ▼
Model Training Module
    │
    ├─ Train/Test Split (80/20)
    ├─ Linear Regression
    └─ Random Forest
    │
    ▼
Model Evaluation
    │
    ├─ R² Score
    ├─ RMSE
    └─ MAE
    │
    ▼
Best Model Selection
    (models/best_model.pkl)
    │
    ▼
Streamlit Web App
    │
    ▼
User Input → Prediction → AQI Category
Feature Engineering Pipeline

Raw weather data → Derived features:

Temperature Range = Max Temp - Min Temp
Captures daily temperature volatility
May influence air mixing/pollution dispersion
Day of Year = 1-365
Captures seasonal patterns
Winter typically has worse air quality
Month = 1-12
Direct monthly seasonality
Different pollution levels by season
🚀 Quick Start
Prerequisites
Python 3.9+
pip
~500MB disk space
Installation (5 minutes)
bash
# 1. Clone repository
git clone https://github.com/usf321/air-quality-prediction.git
cd air-quality-prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
Train Model (Optional - model included)
bash
python src/model.py

Output:

Loading preprocessed data...
Loaded 90 rows, 11 columns
Splitting data into train/test...
Training Linear Regression...
Training Random Forest...

Linear Regression: R² = 0.9711
Random Forest: R² = 0.9656

Best model: Linear Regression
Model saved to models/best_model.pkl
Run Web App
bash
streamlit run app.py

Opens at: http://localhost:8501

Using the App
Adjust weather sliders on the left sidebar
Watch predictions update in real-time
See air quality category (🟢 Good → ⚫ Hazardous)
View input summary table
📁 Project Structure
air-quality-prediction/
│
├── 📁 data/
│   ├── raw/
│   │   └── air_quality_data.csv      (Original API data)
│   └── processed/
│       └── air_quality_clean.csv     (Cleaned & engineered)
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_collector.py             (Fetch from APIs)
│   ├── preprocessing.py              (Clean & engineer)
│   └── model.py                      (Train & evaluate)
│
├── 📁 models/
│   └── best_model.pkl                (Trained model)
│
├── 📁 notebooks/
│   └── 01_exploratory_analysis.ipynb (Data exploration)
│
├── app.py                            (Streamlit web app)
├── requirements.txt                  (Python dependencies)
├── README.md                         (Documentation)
├── .gitignore                        (Git ignore patterns)
└── LICENSE                           (MIT License)
💻 Technologies Used
Data Science & ML
pandas - Data manipulation and analysis
numpy - Numerical computing
scikit-learn - Machine learning models
scipy - Statistical functions
Web Framework
streamlit - Interactive web application
matplotlib - Static visualizations
seaborn - Statistical graphics
APIs & Utilities
requests - HTTP library for API calls
python-dotenv - Environment configuration
🧠 What I Learned
Technical Skills

Data Engineering

Fetching real-world data from REST APIs
Handling missing values and outliers
Feature engineering from raw measurements
Data normalization and scaling

Machine Learning

Model training and evaluation
Hyperparameter tuning concepts
Cross-validation principles
Model selection criteria

Software Development

Project structure best practices
Code organization and modularity
Error handling and logging
Documentation standards

Deployment

Version control with Git
GitHub repository management
Web application deployment
Production-ready code structure
Conceptual Understanding

✅ Why simpler models often outperform complex ones (small dataset)
✅ Importance of train/test splits
✅ Difference between R² and "accuracy"
✅ Why feature engineering matters
✅ How APIs provide real-world data

🔬 Future Research
Short Term (1-2 months)
 Implement proper cross-validation
 Add model explainability (SHAP values)
 Confidence intervals for predictions
 Unit tests and integration tests
 Logging and error tracking
Medium Term (2-4 months)
 Time-series forecasting (24-hour, 7-day ahead)
 LSTM neural network model
 Support multiple cities
 Real-time data pipeline (daily updates)
 Hyperparameter optimization (GridSearch)
Advanced (4+ months)
 XGBoost and LightGBM models
 Ensemble methods
 Docker containerization
 CI/CD pipeline with GitHub Actions
 Cloud deployment (AWS/GCP/Azure)
 REST API endpoint
 Mobile application
 SHAP explainability dashboard
Research Directions
Incorporate satellite imagery (aerosol optical depth)
Add atmospheric pressure and humidity
Model urban heat island effects
Seasonal decomposition (trend + seasonality)
Anomaly detection for sensor malfunctions
📚 Data Sources

All data is fetched from free, open APIs:

Open-Meteo Weather Archive API
Historical weather data
Temperature, precipitation, wind speed
No authentication required
Free tier: 10,000 requests/day
Open-Meteo Air Quality API
Historical air quality measurements
PM2.5, PM10, Ozone concentrations
No authentication required
Coverage: Most global locations

Location Used: Cairo, Egypt (30.04°N, 31.24°E)

💡 Key Insights
Smaller datasets favor simpler models - Linear Regression beat Random Forest here
Feature engineering matters more than algorithm - Time-based features captured seasonality
Clean data beats complex models - Proper preprocessing → good performance
APIs democratize data - No expensive sensors needed for this project
End-to-end pipelines are valuable - Shows practical understanding
📝 Example Usage
Input
PM10:           80 µg/m³
Ozone:          50 ppb
Max Temp:       30°C
Min Temp:       20°C
Precipitation:  0 mm
Wind Speed:     10 m/s
Prediction Output
Predicted PM2.5:  45.3 µg/m³
Category:         🟡 MODERATE
Health Impact:    Acceptable for most. Sensitive groups 
                  (children, elderly, asthmatic) should 
                  limit outdoor activities.
🛠️ Configuration
Change Location

Edit src/data_collector.py:

python
LATITUDE = 30.0444      # Your latitude
LONGITUDE = 31.2357     # Your longitude
LOOKBACK_DAYS = 90      # Historical days
Adjust Model Parameters

Edit src/model.py:

python
test_size=0.2           # 80/20 train/test split
n_estimators=100        # Random Forest trees
max_depth=15            # Maximum tree depth
random_state=42         # Reproducibility
📄 License

MIT License - Free to use for educational, portfolio, and commercial purposes.

MIT License

Copyright (c) 2024 Youssef

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
👤 About

Youssef - High School Student | Machine Learning Enthusiast | Future Computer Science Student

🎓 AI & Data Science Course (In Progress)
🔬 Building ML projects for scholarship applications
💻 Python, Machine Learning, Data Science
🌐 GitHub: @usf321
📞 Questions & Feedback

If you have questions or suggestions:

Read the comments - Code has inline explanations
Check the notebook - notebooks/01_exploratory_analysis.ipynb
Review the code - Well-organized, easy to follow
Open an issue - GitHub Issues for bugs/features
🙏 Acknowledgments
Open-Meteo - Free weather and air quality APIs
Streamlit - Web framework for data apps
Scikit-learn - ML tools and algorithms
Python Community - Libraries and documentation

Version: 1.0
Status: Active Development
Last Updated: July 2026
Maintained: Yes
