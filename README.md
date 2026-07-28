Air Quality Prediction System 

A machine learning project that predicts air pollution (PM2.5) based on weather conditions.

What It Does

Enter weather data (temperature, wind speed, precipitation, etc.) and get real-time air quality predictions with a color-coded safety level.

How to Use
1. Install Requirements
bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
2. Train the Model
bash
python src/model.py
3. Run the Web App
bash
streamlit run app.py

Then open http://localhost:8501 in your browser.

Project Structure
📁 air-quality-prediction/
├ 📁 data/
│   ├── raw/              # Orignal data
│   └── processed/        # Cleaned data
─ 📁 src/
│   ├── data_collector.py    # Fetch data from API
│   ├── preprocessing.py     # Clean & prepare data
│   └── model.py            # Train ML models
├── 📁 models/
│   └── best_model.pkl      # Trained model
├── app.py                  # Web app
├── requirements.txt        # Dependencies
└── README.md
Results
Model: Linear Regression
Accuracy: R² = 0.97 (97% accurate)
Error: ±0.76 µg/m³
Technologies Used
Python
Pandas (data processing)
Scikit-learn (machine learning)
Streamlit (web app)
What I Learned

✅ Building complete ML pipelines
✅ Data cleaning & feature engineering
✅ Model training & evaluation
✅ Deploying web applications

Next Steps
Deploy to Streamlit Cloud for public access
Add more cities/regions
Implement real-time data updates
Add forecast capabilities
