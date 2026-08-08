"""
Air Quality Data Collector

This module fetches historical weather and air quality data from free APIs.
No authentication required.

Usage:
    python src/data_collector.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_air_quality_data(latitude, longitude, days_back=90):
    """
    Fetch historical air quality data from Open-Meteo API.
    
    This function retrieves daily maximum concentrations of pollutants
    like PM2.5, PM10, and Ozone for a specified location.
    
    Args:
        latitude (float): Location latitude (e.g., 30.0444 for Cairo)
        longitude (float): Location longitude (e.g., 31.2357 for Cairo)
        days_back (int): Number of historical days to fetch (default: 90, max: 90)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - date: Date of measurement
            - pm2_5_max: Daily max PM2.5 in µg/m³
            - pm10_max: Daily max PM10 in µg/m³
            - o3_max: Daily max Ozone in ppb
        Returns None if API call fails
    
    Example:
        >>> df = fetch_air_quality_data(30.0444, 31.2357, days_back=90)
        >>> print(df.shape)
        (90, 4)
    """
    
    # Calculate date range (from days_back ago to today)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    # Open-Meteo Air Quality API endpoint (free, no API key needed)
    base_url = "https://air-quality-api.open-meteo.com/v1/air_quality"
    
    # API parameters: location, date range, and pollutants to fetch
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "pm2_5,pm10,o3,no2,so2,co",  # Hourly pollutants
        "daily": "pm2_5_max,pm10_max,o3_max",  # Daily maximums (what we use)
        "timezone": "auto"
    }
    
    print(f"Fetching air quality data for ({latitude}, {longitude}) - {days_back} days...")
    
    try:
        # Make HTTP request to the API
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise error if request failed
        data = response.json()
        
        # Extract daily data and convert to DataFrame
        daily_data = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "pm2_5_max": data["daily"]["pm2_5_max"],
            "pm10_max": data["daily"]["pm10_max"],
            "o3_max": data["daily"]["o3_max"],
        })
        
        print(f"✓ Successfully fetched {len(daily_data)} days of air quality data")
        return daily_data
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching air quality data: {e}")
        return None


def fetch_weather_data(latitude, longitude, days_back=90):
    """
    Fetch historical weather data from Open-Meteo Archive API.
    
    Retrieves daily weather metrics including temperature, precipitation,
    and wind speed for a specified location.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        days_back (int): Number of historical days to fetch (default: 90)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - date: Date of measurement
            - temp_max: Daily maximum temperature in °C
            - temp_min: Daily minimum temperature in °C
            - precipitation: Daily precipitation in mm
            - wind_speed: Daily max wind speed in m/s
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
    
    print(f"Fetching weather data for ({latitude}, {longitude}) - {days_back} days...")
    
    try:
        response = requests.get(base_url, params=params)
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


def combine_data(air_quality_df, weather_df):
    """
    Combine air quality and weather data into a single DataFrame.
    
    Merges two DataFrames on the 'date' column using an inner join,
    keeping only dates present in both datasets.
    
    Args:
        air_quality_df (pd.DataFrame): Air quality data from fetch_air_quality_data()
        weather_df (pd.DataFrame): Weather data from fetch_weather_data()
    
    Returns:
        pd.DataFrame: Combined dataset with all features
    
    Example:
        >>> combined = combine_data(air_quality_df, weather_df)
        >>> print(combined.columns)
        Index(['date', 'pm2_5_max', 'pm10_max', 'o3_max', 'temp_max', ...])
    """
    
    combined = pd.merge(air_quality_df, weather_df, on="date", how="inner")
    print(f"✓ Combined dataset created: {len(combined)} records")
    return combined


def save_data(df, filepath):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path where to save the CSV file
    """
    
    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to {filepath}")


if __name__ == "__main__":
    # ============================================
    # Configuration: Set your location here
    # ============================================
    LATITUDE = 30.0444      # Cairo, Egypt latitude
    LONGITUDE = 31.2357     # Cairo, Egypt longitude
    DAYS_BACK = 1095        # 3 years of historical data (365 * 3)
    
    print("🌍 Air Quality Data Collection Pipeline")
    print("=" * 50)
    print(f"Collecting {DAYS_BACK} days of data...")
    
    # Step 1: Fetch air quality data
    air_quality = fetch_air_quality_data(LATITUDE, LONGITUDE, days_back=DAYS_BACK)
    
    # Step 2: Fetch weather data
    weather = fetch_weather_data(LATITUDE, LONGITUDE, days_back=DAYS_BACK)
    
    # Step 3: Combine and save if both fetches succeeded
    if air_quality is not None and weather is not None:
        combined = combine_data(air_quality, weather)
        save_data(combined, "data/raw/air_quality_data.csv")
        print("\n" + "=" * 50)
        print("✓ Data collection complete!")
        print(f"Total records: {len(combined)}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ Failed to fetch data. Check your internet connection.")
        print("=" * 50)
