# Air Quality Prediction System

A machine learning pipeline predicting PM2.5 levels and providing results in Streamlit dashboard.

## Overview

The project constructs a system for air quality prediction based on weather data. The system processes gathered information and trains a Linear Regression and Random Forests model, assessing their predictive performance.

The model achieved an R² value of 0.97 on the validation sample using Random Forest method. Additionally, the system provides a live prediction dashboard made with Streamlit library, processes temperature range and day of the year values, and handles API or file related errors.

---

## 🖼️ Dashboard Screenshots

**Screenshot 1: Home Page & Main Prediction**
<img width="1568" height="751" alt="Solid_black svg" src="https://github.com/user-attachments/assets/478f227b-50f4-4316-a9d9-88f67a36107b" />


**Screenshot 2: Weather Input & Parameters**
![Input Summary](./screenshots/input_summary.png)

**Screenshot 3: Model Information & How It Works**
![Model Info](./screenshots/model_info.png)

**Screenshot 4: Live App Running**
![Live App](./screenshots/live_app.png)

---

## How it is structured

The project directory is organized as follows:

├── .github/workflows/ # CI actions
├── data/ # historical data
├── models/ # trained weights (best_model.pkl)
├── src/ # data preprocessing and training
├── app.py # dashboard app
├── README.md
└── requirements.txt

...rest of README
