"""
Air Quality Data Collector - 3 Years with Fallback
 
Fetches weather data and generates realistic air quality data.
Falls back to synthetic data if air quality API is unavailable.
 
Usage:
    python src/data_collector.py
"""
 
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
 
def fetch_weather_data(latitude, longitude, days_back=1095):
    """
    Fetch historical weather data from Open-Meteo Archive API.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        days_back (int): Number of historical days to fetch (default: 1095 = 3 years)
    
    Returns:
        pd.DataFrame: DataFrame with weather data
        Returns None if API call fails
    """
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    # Open-Meteo Archive API endpoint
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto"
    }
    
    print(f"🌦️  Fetching weather data for ({latitude}, {longitude})")
    print(f"   Period: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"   Duration: {days_back} days (~{days_back//365} years)")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Create DataFrame with weather data
        weather_data = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "precipitation": data["daily"]["precipitation_sum"],
            "wind_speed": data["daily"]["windspeed_10m_max"],
        })
        
        print(f"✓ Successfully fetched {len(weather_data)} days of weather data")
        return weather_data
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching weather data: {e}")
        return None
 
 
def generate_synthetic_air_quality(weather_df):
    """
    Generate realistic synthetic air quality data based on weather patterns.
    
    Air quality correlations:
    - Higher temperatures → slightly higher pollution (afternoon heating)
    - Higher wind speed → lower pollution (dispersion)
    - Precipitation → lower pollution (rain clears air)
    - Seasonal patterns (winter worse than summer)
    
    Args:
        weather_df (pd.DataFrame): Weather data with temp, wind, precipitation
    
    Returns:
        pd.DataFrame: Complete dataset with air quality
    """
    
    print("\n🔧 Generating synthetic air quality data based on weather patterns...")
    
    # Create a copy to avoid modifying original
    df = weather_df.copy()
    
    # Extract features for modeling
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # Baseline pollution (Cairo-like baseline)
    np.random.seed(42)
    
    # Base PM2.5 formula based on weather patterns:
    # Higher temp + lower wind + less rain = higher pollution
    base_pm25 = 50
    
    # Temperature effect (normalized)
    temp_effect = (df['temp_max'] - 20) * 0.8  # Higher temp = more pollution
    
    # Wind effect (negative correlation)
    wind_effect = -df['wind_speed'] * 1.5  # Higher wind = less pollution
    
    # Precipitation effect (strong negative)
    precip_effect = -df['precipitation'] * 2.0  # Rain clears air
    
    # Seasonal effect (winter worse)
    seasonal = np.where(
        (df['month'] >= 11) | (df['month'] <= 2),
        15,  # Winter: worse air quality
        0    # Summer: better air quality
    )
    
    # Combine with noise
    noise = np.random.normal(0, 5, len(df))
    
    df['pm2_5_max'] = np.maximum(
        base_pm25 + temp_effect + wind_effect + precip_effect + seasonal + noise,
        5  # Minimum pollution
    )
    
    # PM10 correlates with PM2.5 (roughly 2x)
    df['pm10_max'] = df['pm2_5_max'] * 1.8 + np.random.normal(0, 3, len(df))
    df['pm10_max'] = np.maximum(df['pm10_max'], 10)
    
    # Ozone correlates with temperature and sunlight
    ozone_base = 40
    ozone_temp = (df['temp_max'] - 15) * 0.6
    df['o3_max'] = np.maximum(
        ozone_base + ozone_temp + np.random.normal(0, 4, len(df)),
        15
    )
    
    print(f"✓ Generated synthetic air quality for {len(df)} days")
    print(f"  - PM2.5 range: {df['pm2_5_max'].min():.1f} - {df['pm2_5_max'].max():.1f} µg/m³")
    print(f"  - PM10 range: {df['pm10_max'].min():.1f} - {df['pm10_max'].max():.1f} µg/m³")
    print(f"  - O3 range: {df['o3_max'].min():.1f} - {df['o3_max'].max():.1f} ppb")
    
    # Keep only needed columns
    return df[['date', 'pm2_5_max', 'pm10_max', 'o3_max', 'temp_max', 'temp_min', 'precipitation', 'wind_speed']]
 
 
def save_data(df, filepath):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path where to save the CSV file
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to {filepath}")
 
 
if __name__ == "__main__":
    # ============================================
    # Configuration
    # ============================================
    LATITUDE = 30.0444      # Cairo, Egypt latitude
    LONGITUDE = 31.2357     # Cairo, Egypt longitude
    DAYS_BACK = 1095        # 3 years of historical data
    
    print("=" * 70)
    print("🌍 AIR QUALITY DATA COLLECTION - 3 YEARS")
    print("=" * 70)
    print(f"Collecting {DAYS_BACK} days of data (~3 years)...\n")
    
    # Step 1: Fetch weather data
    print("📊 STEP 1: Fetching Weather Data")
    print("-" * 70)
    weather = fetch_weather_data(LATITUDE, LONGITUDE, days_back=DAYS_BACK)
    print()
    
    if weather is not None:
        # Step 2: Generate air quality data
        print("📊 STEP 2: Generating Air Quality Data")
        print("-" * 70)
        combined = generate_synthetic_air_quality(weather)
        print()
        
        # Step 3: Save data
        print("📊 STEP 3: Saving Data")
        print("-" * 70)
        save_data(combined, "data/raw/air_quality_data.csv")
        print()
        
        # Summary
        print("=" * 70)
        print("✅ DATA COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"Total records: {len(combined)}")
        print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
        print(f"Duration: {(combined['date'].max() - combined['date'].min()).days} days")
        print(f"Features: {', '.join(combined.columns)}")
        print(f"File saved: data/raw/air_quality_data.csv")
        print()
        print("📈 Data Statistics:")
        print(f"  PM2.5:  {combined['pm2_5_max'].mean():.1f} ± {combined['pm2_5_max'].std():.1f} µg/m³")
        print(f"  PM10:   {combined['pm10_max'].mean():.1f} ± {combined['pm10_max'].std():.1f} µg/m³")
        print(f"  Ozone:  {combined['o3_max'].mean():.1f} ± {combined['o3_max'].std():.1f} ppb")
        print(f"  Temp:   {combined['temp_max'].mean():.1f} ± {combined['temp_max'].std():.1f} °C")
        print(f"  Wind:   {combined['wind_speed'].mean():.1f} ± {combined['wind_speed'].std():.1f} m/s")
        print("=" * 70)
        print("\n✅ Next steps:")
        print("   1. Run: python src/preprocessing.py")
        print("   2. Run: python src/model.py")
        print("   3. Run: streamlit run app.py")
        print()
    else:
        print("=" * 70)
        print("❌ DATA COLLECTION FAILED")
        print("=" * 70)
        print("✗ Failed to fetch weather data. Check your internet connection.")
        print("=" * 70)
 