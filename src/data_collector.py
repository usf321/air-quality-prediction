"""
Air Quality Data Collector

Fetches historical weather and air quality data from free APIs.
Used to build training datasets for ML models.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_air_quality_data(latitude, longitude, days_back=90):
    """
    Get air quality measurements from Open-Meteo API.
    
    Returns daily max values for PM2.5, PM10, and Ozone.
    These are the main pollutants we care about.
    """
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    base_url = "https://air-quality-api.open-meteo.com/v1/air_quality"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "pm2_5_max,pm10_max,o3_max",
        "timezone": "auto"
    }
    
    print(f"Fetching air quality data for ({latitude}, {longitude})...")
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "pm2_5_max": data["daily"]["pm2_5_max"],
            "pm10_max": data["daily"]["pm10_max"],
            "o3_max": data["daily"]["o3_max"],
        })
        
        print(f"Got {len(df)} days of air quality data")
        return df
    
    except requests.exceptions.Timeout:
        print("Request timed out. Try again in a moment.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection failed. Check your internet.")
        return None
    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


def fetch_weather_data(latitude, longitude, days_back=90):
    """
    Get weather data from Open-Meteo Archive API.
    
    We need: temperature, wind speed, and precipitation.
    These influence how pollution spreads in the air.
    """
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto"
    }
    
    print(f"Fetching weather data for ({latitude}, {longitude})...")
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather_df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "precipitation": data["daily"]["precipitation_sum"],
            "wind_speed": data["daily"]["windspeed_10m_max"],
        })
        
        print(f"Got {len(weather_df)} days of weather data")
        return weather_df
    
    except Exception as e:
        print(f"Weather data fetch failed: {e}")
        return None


def combine_datasets(air_quality, weather):
    """Merge air quality and weather into one dataset."""
    
    if air_quality is None or weather is None:
        return None
    
    merged = pd.merge(air_quality, weather, on="date", how="inner")
    print(f"Merged into {len(merged)} complete records")
    return merged


def save_raw_data(df, output_path):
    """Save the combined dataset to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def main():
    # Configuration
    CAIRO_LAT = 30.0444
    CAIRO_LON = 31.2357
    LOOKBACK_DAYS = 90
    
    print("Starting data collection...")
    
    # Fetch both datasets
    air_data = fetch_air_quality_data(CAIRO_LAT, CAIRO_LON, LOOKBACK_DAYS)
    weather_data = fetch_weather_data(CAIRO_LAT, CAIRO_LON, LOOKBACK_DAYS)
    
    # Combine if both succeeded
    if air_data is not None and weather_data is not None:
        combined_data = combine_datasets(air_data, weather_data)
        if combined_data is not None:
            save_raw_data(combined_data, "data/raw/air_quality_data.csv")
            print("Data collection completed successfully")
    else:
        print("Failed to fetch data. Check your connection.")


if __name__ == "__main__":
    main()