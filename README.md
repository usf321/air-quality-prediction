Air Quality Prediction System

A machine learning pipeline predicting PM2,5 levels and providing results in Streamlit dashboard.

Overview

The project constructs a system for air quality prediction based on weather data. The system processes gathered information and trains a Linear Regression and Random Forests model, assessing their predictive performance.

The model achieved an R² value of 0,97 on the validation sample using Random Forest method. Additionally, the system provides a live prediction dashboard made with Streamlit library, processes temperature range and day of the year values, and handles API or file related errors.

How it is structured
The project directory is organized as follows:

├── .github/workflows/ # CI actions
├── data/ # historical data
├── models/ # trained weights (best_model.pkl)
├── src/ # data preprocessing and training
├── app.py # dashboard app
├── README.md
└── requirements.txt

Model Performance

Linear Regression
Random Forest
Training R² 0,95 0,98
Validation R² 0,94 0,97
MAE, µg/m³ ±1,42 ±0,76

Note: On both data samples, the Random Forest demonstrated its ability to generalize better than Linear Regression achieving nearly two times lower prediction error.

Getting Started

To run the application, first, clone this repository and install the required dependencies:

git clone https://github.com
cd air-quality-prediciton
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt

Then, train the model

python src/model.py

Finally, launch the streamlit dashboard

streamlit run app.py
