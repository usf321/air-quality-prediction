# Air Quality Prediction System

A machine learning pipeline predicting PM2.5 air pollution levels based on weather data.

## Overview

This project builds an end-to-end ML system that predicts PM2.5 using weather conditions. It includes data collection, preprocessing, model training, and an interactive Streamlit dashboard.

**Model Performance:** R² = 0.97 (Linear Regression)

## 🖼️ Dashboard Screenshots

**Prediction Interface**
![Preview 1](https://github.com/user-attachments/assets/ef71ad35-7f4b-4969-8517-a128b48dfee6)

**Weather Parameters**
![Preview 2](https://github.com/user-attachments/assets/c17687c7-1114-49fe-8b23-7dae4a6ef759)

**Model Information**
![Preview 3](https://github.com/user-attachments/assets/1773de0d-9821-4b6b-9d9d-af6fe91aa282)

**Live App Running**
![Preview 4](https://github.com/user-attachments/assets/478f227b-50f4-4316-a9d9-88f67a36107b)

## Quick Start

```bash
git clone https://github.com/usf321/air-quality-prediction.git
cd air-quality-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/model.py
streamlit run app.py
```

## Project Structure

├── data/ # Raw & processed data
├── src/ # Data collection, preprocessing, model training
├── models/ # Trained model (best_model.pkl)
├── app.py # Streamlit dashboard
├── requirements.txt
└── README.md


## Model Performance

| Metric | Linear Regression |
|--------|-------------------|
| R² Score | 0.9711 |
| RMSE | 0.7586 µg/m³ |
| MAE | 0.4379 µg/m³ |

## What It Does

1. Collects weather & air quality data from free APIs
2. Cleans data and engineers 3 features (temperature range, day of year, month)
3. Trains Linear Regression model (R²=0.97)
4. Provides interactive Streamlit dashboard
5. Makes real-time PM2.5 predictions

## Technologies

- Python, pandas, scikit-learn, Streamlit
- Open-Meteo APIs (weather & air quality data)

---

**Version:** 1.1 | **Status:** Active Development
