"""
Data Preprocessing Module

Cleans the raw data and prepares it for model training.
Handles missing values, removes outliers, and creates new features.
"""

import pandas as pd
import numpy as np
from scipy import stats


class DataCleaner:
    """Handles all data cleaning operations."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.initial_rows = 0
    
    def load(self):
        """Read CSV and parse dates."""
        self.df = pd.read_csv(self.filepath)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.initial_rows = len(self.df)
        print(f"Loaded {self.initial_rows} rows")
        return self
    
    def clean_missing_values(self):
        """Remove rows with any NaN values."""
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        removed = before - after
        if removed > 0:
            print(f"Removed {removed} rows with missing values")
        return self
    
    def clean_outliers(self):
        """
        Remove outliers using z-score.
        Anything beyond 3 standard deviations is removed.
        """
        pollution_cols = ["pm2_5_max", "pm10_max", "o3_max"]
        
        before = len(self.df)
        z_scores = np.abs(stats.zscore(self.df[pollution_cols]))
        self.df = self.df[(z_scores < 3).all(axis=1)]
        after = len(self.df)
        removed = before - after
        
        if removed > 0:
            print(f"Removed {removed} outlier rows")
        return self
    
    def engineer_features(self):
        """Create new features from existing ones."""
        # How much temperature varies in a day
        self.df["temp_range"] = self.df["temp_max"] - self.df["temp_min"]
        
        # What day of the year (1-365)
        self.df["day_of_year"] = self.df["date"].dt.dayofyear
        
        # What month (1-12)
        self.df["month"] = self.df["date"].dt.month
        
        print("Added 3 new features")
        return self
    
    def save(self, output_path):
        """Save cleaned data to CSV."""
        self.df.to_csv(output_path, index=False)
        print(f"Saved {len(self.df)} rows to {output_path}")
        return self
    
    def summary(self):
        """Print summary of what was done."""
        removed = self.initial_rows - len(self.df)
        print(f"\nSummary:")
        print(f"  Started with: {self.initial_rows} rows")
        print(f"  Removed: {removed} rows")
        print(f"  Final: {len(self.df)} rows")
        print(f"  Columns: {len(self.df.columns)}")


def main():
    print("Starting data preprocessing...\n")
    
    cleaner = DataCleaner("data/raw/air_quality_data.csv")
    
    (cleaner
     .load()
     .clean_missing_values()
     .clean_outliers()
     .engineer_features()
     .save("data/processed/air_quality_clean.csv")
     .summary())
    
    print("\nDone!")


if __name__ == "__main__":
    main()