# Air Quality Prediction System

A machine learning pipeline predicting PM2.5 levels and providing results in Streamlit dashboard.

## Overview

The project constructs a system for air quality prediction based on weather data. The system processes gathered information and trains a Linear Regression and Random Forests model, assessing their predictive performance.

The model achieved an R² value of 0.97 on the validation sample using Random Forest method. Additionally, the system provides a live prediction dashboard made with Streamlit library, processes temperature range and day of the year values, and handles API or file related errors.

---

## 🖼️ Dashboard Screenshots

**Screenshot 1: Home Page & Main Prediction**

<img width="1568" height="743" alt="preview (2)" src="https://github.com/user-attachments/assets/ef71ad35-7f4b-4969-8517-a128b48dfee6" />


**Screenshot 2: Weather Input & Parameters**

<img width="1568" height="729" alt="preview (1)" src="https://github.com/user-attachments/assets/c17687c7-1114-49fe-8b23-7dae4a6ef759" />


**Screenshot 3: Model Information & How It Works**

<img width="1568" height="744" alt="preview" src="https://github.com/user-attachments/assets/1773de0d-9821-4b6b-9d9d-af6fe91aa282" />


**Screenshot 4: Live App Running**

<img width="1568" height="751" alt="Solid_black svg" src="https://github.com/user-attachments/assets/478f227b-50f4-4316-a9d9-88f67a36107b" />

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
