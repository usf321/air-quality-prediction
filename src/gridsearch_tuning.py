"""
GridSearchCV Hyperparameter Tuning for Air Quality Model

Optimize hyperparameters for Ridge, Lasso, and ElasticNet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import time


def load_data():
    """Load processed data"""
    df = pd.read_csv("data/processed/air_quality_clean.csv")
    
    features = [
        "pm10_max",
        "o3_max",
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed",
        "temp_range",
        "day_of_year",
        "month",
    ]
    
    X = df[features]
    y = df["pm2_5_max"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def tune_ridge(X_train, y_train, X_test, y_test):
    """Tune Ridge hyperparameters"""
    print("\n" + "=" * 70)
    print("TUNING RIDGE REGRESSION")
    print("=" * 70)
    
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    grid_search = GridSearchCV(
        Ridge(),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_["alpha"]
    
    pred = best_model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Time: {elapsed:.2f}s")
    
    return {
        "model": best_model,
        "alpha": best_alpha,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def tune_lasso(X_train, y_train, X_test, y_test):
    """Tune Lasso hyperparameters"""
    print("\n" + "=" * 70)
    print("TUNING LASSO REGRESSION")
    print("=" * 70)
    
    param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    
    grid_search = GridSearchCV(
        Lasso(max_iter=10000),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_["alpha"]
    
    pred = best_model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Time: {elapsed:.2f}s")
    
    return {
        "model": best_model,
        "alpha": best_alpha,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def tune_elasticnet(X_train, y_train, X_test, y_test):
    """Tune ElasticNet hyperparameters"""
    print("\n" + "=" * 70)
    print("TUNING ELASTICNET")
    print("=" * 70)
    
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8],
    }
    
    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_["alpha"]
    best_l1 = grid_search.best_params_["l1_ratio"]
    
    pred = best_model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best l1_ratio: {best_l1}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Time: {elapsed:.2f}s")
    
    return {
        "model": best_model,
        "alpha": best_alpha,
        "l1_ratio": best_l1,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    print(f"\nData loaded: {len(X_train)} training, {len(X_test)} test")
    
    # Tune models
    ridge_results = tune_ridge(X_train, y_train, X_test, y_test)
    lasso_results = tune_lasso(X_train, y_train, X_test, y_test)
    elasticnet_results = tune_elasticnet(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    
    results = {
        "Ridge": ridge_results,
        "Lasso": lasso_results,
        "ElasticNet": elasticnet_results,
    }
    
    print("\nModel Comparison (Test Set):")
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 45)
    
    for name, result in results.items():
        print(f"{name:<15} {result['r2']:<10.4f} {result['rmse']:<10.4f} {result['mae']:<10.4f}")
    
    # Save best models
    with open("models/tuned_ridge.pkl", "wb") as f:
        pickle.dump(ridge_results["model"], f)
    
    with open("models/tuned_lasso.pkl", "wb") as f:
        pickle.dump(lasso_results["model"], f)
    
    with open("models/tuned_elasticnet.pkl", "wb") as f:
        pickle.dump(elasticnet_results["model"], f)
    
    print("\n✅ Tuned models saved!")
    print("=" * 70)
