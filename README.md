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

Finally, launch the streamlit d

versioun 1.0
---
What's New in v1.1

- **Stacking Ensemble Model** - Combined 4 base learners (Linear, Ridge, Lasso, ElasticNet) with meta-learner
- **Improved Performance** - R² = 0.9715 (outperforms all individual models)
- **Extended Dataset** - 1,091 days (~3 years) of historical weather data
- **Production Ready** - Deployed live with interactive web UI


**Learned Meta-Weights:**
- Linear: 0.25
- Ridge: 0.20
- Lasso: 0.30
- ElasticNet: 0.25

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train stacking model
python src/stacking_model.py

# Run web app
streamlit run app.py
```

**v1.1** - Stacking Ensemble (Current)
- Added 4-model stacking with meta-learner
- Extended to 1,091 days data
- R² improved to 0.9715

**v1.0** - Single Model
- Individual model implementations
- 90 days initial data
- R² = 0.9709
