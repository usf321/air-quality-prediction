"""
Model Training Module

Trains two different ML models and picks the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


class ModelTrainer:
    """Manages model training and evaluation."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load processed data."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records")
        return self
    
    def prepare_features(self):
        """Split features and target."""
        features = [
            "pm10_max", "o3_max",
            "temp_max", "temp_min", 
            "precipitation", "wind_speed",
            "temp_range", "day_of_year", "month"
        ]
        
        X = self.df[features]
        y = self.df["pm2_5_max"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        return self
    
    def train_linear_model(self):
        """Train simple linear regression."""
        print("\nTraining Linear Regression...")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models["linear"] = model
        
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        
        self.results["linear"] = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "model": model
        }
        
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        return self
    
    def train_forest_model(self):
        """Train random forest."""
        print("\nTraining Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models["forest"] = model
        
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        
        self.results["forest"] = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "model": model
        }
        
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        return self
    
    def pick_best(self):
        """Choose the model with highest R²."""
        best_name = max(self.results, key=lambda x: self.results[x]["r2"])
        best_model = self.results[best_name]["model"]
        
        print(f"\nBest model: {best_name.upper()}")
        return best_model
    
    def save_model(self, model, path):
        """Save trained model to disk."""
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
    
    def show_comparison(self):
        """Print model comparison table."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        for name, metrics in self.results.items():
            print(f"\n{name.upper()}:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")


def main():
    trainer = ModelTrainer("data/processed/air_quality_clean.csv")
    
    (trainer
     .load_data()
     .prepare_features()
     .train_linear_model()
     .train_forest_model())
    
    trainer.show_comparison()
    
    best_model = trainer.pick_best()
    trainer.save_model(best_model, "models/best_model.pkl")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()