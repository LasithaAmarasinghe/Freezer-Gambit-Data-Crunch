import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the components
from deployment.data_pipeline import DataPipeline
from deployment.feature_engineering import FeatureEngineer
from deployment.model_trainer import ModelTrainer

def train_models(commodity=None, region=None, model_type='xgboost', n_trials=20):
    """
    Train models for all commodities or a specific commodity and region.

    Args:
        commodity (str, optional): Specific commodity to train for
        region (str, optional): Specific region to train for
        model_type (str): Type of model to train ('xgboost' or 'lightgbm')
        n_trials (int): Number of Optuna trials
    """
    # Define paths
    PRICE_DATA_PATH = os.path.join("data", "price_data.csv")
    WEATHER_DATA_PATH = os.path.join("data", "weather_data.csv")
    MODELS_DIR = os.path.join("models")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(PRICE_DATA_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

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

    # Initialize components
    print("Initializing components...")
    data_pipeline = DataPipeline(PRICE_DATA_PATH, WEATHER_DATA_PATH)
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer(MODELS_DIR)

    # Get merged data
    print("Loading and merging data...")
    merged_data = data_pipeline.get_merged_data()

    # Apply feature engineering
    print("Applying feature engineering...")
    start_time = time.time()

    if commodity:
        # Filter data for the specific commodity
        filtered_data = merged_data[merged_data['commodity'] == commodity]

        if region:
            # Filter data for the specific region
            filtered_data = filtered_data[filtered_data['region'] == region]

        if filtered_data.empty:
            print(f"No data available for {commodity} in {region}")
            return

        featured_data = feature_engineer.engineer_features(filtered_data)
    else:
        featured_data = feature_engineer.engineer_features(merged_data)

    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    print(f"Featured data shape: {featured_data.shape}")

    # Train models
    if commodity:
        print(f"\nTraining model for {commodity} in {region if region else 'all regions'}...")
        start_time = time.time()

        model, params, feature_cols, metrics = model_trainer.train_model_for_commodity(
            featured_data, commodity, region, model_type, n_trials
        )

        if model is not None:
            model_trainer.save_model(model, commodity, region, model_type, params, feature_cols, metrics)
            print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        else:
            print(f"Failed to train model for {commodity} in {region if region else 'all regions'}")
    else:
        print("\nTraining models for all commodities...")
        start_time = time.time()

        models = model_trainer.train_all_commodity_models(
            featured_data, model_type=model_type, n_trials=n_trials, by_region=bool(region)
        )

        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        print(f"Trained {len(models)} models")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train price forecasting models')
    parser.add_argument('--commodity', type=str, help='Specific commodity to train for')
    parser.add_argument('--region', type=str, help='Specific region to train for')
    parser.add_argument('--model-type', type=str, default='xgboost', choices=['xgboost', 'lightgbm'], help='Type of model to train')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')

    args = parser.parse_args()

    train_models(args.commodity, args.region, args.model_type, args.n_trials)
