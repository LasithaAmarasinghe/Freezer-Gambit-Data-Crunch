import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define paths for data files
PRICE_DATA_PATH = os.path.join("data", "price_data.csv")
WEATHER_DATA_PATH = os.path.join("data", "weather_data.csv")

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(PRICE_DATA_PATH), exist_ok=True)

# Check if data files exist, if not, copy from Datasets directory
if not os.path.exists(PRICE_DATA_PATH):
    source_path = os.path.join("..", "Datasets", "PriceData", "train_data.csv")
    if os.path.exists(source_path):
        print(f"Copying price data from {source_path} to {PRICE_DATA_PATH}")
        df = pd.read_csv(source_path)
        df.to_csv(PRICE_DATA_PATH, index=False)
    else:
        print(f"Warning: Source price data file {source_path} not found")

if not os.path.exists(WEATHER_DATA_PATH):
    source_path = os.path.join("..", "Datasets", "WeatherData", "train_data.csv")
    if os.path.exists(source_path):
        print(f"Copying weather data from {source_path} to {WEATHER_DATA_PATH}")
        df = pd.read_csv(source_path)
        df.to_csv(WEATHER_DATA_PATH, index=False)
    else:
        print(f"Warning: Source weather data file {source_path} not found")

# Run the application
if __name__ == "__main__":
    print("Starting AgroChill Price Forecasting API...")
    uvicorn.run("deployment.api:app", host="0.0.0.0", port=8000, reload=True)
