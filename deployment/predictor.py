import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import xgboost as xgb
import lightgbm as lgb

class Predictor:
    """
    Advanced predictor for crop price forecasting.
    Combines the best approaches from multiple solutions.
    """
    
    def __init__(self, data_pipeline, feature_engineer, model_trainer):
        """
        Initialize the Predictor.
        
        Args:
            data_pipeline: Data pipeline instance
            feature_engineer: Feature engineer instance
            model_trainer: Model trainer instance
        """
        self.data_pipeline = data_pipeline
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        
        # Cache for loaded models
        self.model_cache = {}
    
    def get_model(self, commodity, region=None):
        """
        Get a trained model for a specific commodity and region.
        Uses caching to avoid reloading models.
        
        Args:
            commodity (str): Commodity to get model for
            region (str, optional): Region to get model for
        
        Returns:
            tuple: Model, model type, feature columns, and metrics
        """
        # Create cache key
        cache_key = (commodity, region)
        
        # Check if model is in cache
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Try to load model with region
        if region is not None:
            model, model_type, feature_cols, metrics = self.model_trainer.load_model(commodity, region)
            if model is not None:
                self.model_cache[cache_key] = (model, model_type, feature_cols, metrics)
                return model, model_type, feature_cols, metrics
        
        # If no region-specific model or it failed, try commodity-only model
        model, model_type, feature_cols, metrics = self.model_trainer.load_model(commodity)
        
        # Cache model
        if model is not None:
            self.model_cache[(commodity, None)] = (model, model_type, feature_cols, metrics)
        
        return model, model_type, feature_cols, metrics
    
    def predict_prices(self, commodity, region, num_weeks=4):
        """
        Predict prices for a specific commodity and region.
        
        Args:
            commodity (str): Commodity to predict for
            region (str): Region to predict for
            num_weeks (int): Number of weeks to predict
        
        Returns:
            list: List of (date, price) tuples for the predicted weeks
        """
        # Get historical data for the commodity and region
        historical_data = self.data_pipeline.get_data_for_commodity_region(commodity, region)
        
        if historical_data.empty:
            print(f"No historical data available for {commodity} in {region}")
            return []
        
        # Get the latest date in the data
        latest_date = historical_data['date'].max()
        
        # Generate future dates
        prediction_dates = [
            latest_date + timedelta(days=7 * (i + 1))
            for i in range(num_weeks)
        ]
        
        # First try to get a region-specific model
        model, model_type, feature_cols, metrics = self.get_model(commodity, region)
        
        if model is None:
            print(f"No model available for {commodity}")
            return []
        
        # Prepare features for prediction
        future_features = self.feature_engineer.prepare_features_for_prediction(
            historical_data, prediction_dates=prediction_dates
        )
        
        if future_features.empty:
            print(f"Failed to generate features for prediction")
            return []
        
        # Make predictions
        predictions = self.model_trainer.predict(model, model_type, future_features, feature_cols)
        
        if predictions is None:
            print(f"Failed to generate predictions")
            return []
        
        # Create result list
        result = []
        for i, date in enumerate(prediction_dates):
            # Ensure predictions are non-negative
            price = max(0, predictions[i])
            result.append((date, price))
        
        return result
    
    def predict_with_ensemble(self, commodity, region, num_weeks=4, ensemble_size=5):
        """
        Predict prices using an ensemble of models.
        
        Args:
            commodity (str): Commodity to predict for
            region (str): Region to predict for
            num_weeks (int): Number of weeks to predict
            ensemble_size (int): Number of models in the ensemble
        
        Returns:
            tuple: List of (date, price) tuples and confidence intervals
        """
        # Get historical data for the commodity and region
        historical_data = self.data_pipeline.get_data_for_commodity_region(commodity, region)
        
        if historical_data.empty:
            print(f"No historical data available for {commodity} in {region}")
            return [], []
        
        # Get the latest date in the data
        latest_date = historical_data['date'].max()
        
        # Generate future dates
        prediction_dates = [
            latest_date + timedelta(days=7 * (i + 1))
            for i in range(num_weeks)
        ]
        
        # Get model
        model, model_type, feature_cols, metrics = self.get_model(commodity, region)
        
        if model is None:
            print(f"No model available for {commodity}")
            return [], []
        
        # Prepare features for prediction
        future_features = self.feature_engineer.prepare_features_for_prediction(
            historical_data, prediction_dates=prediction_dates
        )
        
        if future_features.empty:
            print(f"Failed to generate features for prediction")
            return [], []
        
        # Create ensemble predictions
        ensemble_predictions = []
        
        for _ in range(ensemble_size):
            # Add random noise to features to create ensemble diversity
            noisy_features = future_features.copy()
            for col in noisy_features.columns:
                if col not in ['date', 'region', 'commodity', 'type', 'price']:
                    # Add small random noise (1-2% of the standard deviation)
                    if noisy_features[col].std() > 0:
                        noise = np.random.normal(0, 0.01 * noisy_features[col].std(), size=len(noisy_features))
                        noisy_features[col] = noisy_features[col] + noise
            
            # Make predictions with the noisy features
            predictions = self.model_trainer.predict(model, model_type, noisy_features, feature_cols)
            
            if predictions is not None:
                ensemble_predictions.append(predictions)
        
        if not ensemble_predictions:
            print(f"Failed to generate ensemble predictions")
            return [], []
        
        # Calculate mean and confidence intervals
        ensemble_array = np.array(ensemble_predictions)
        mean_predictions = np.mean(ensemble_array, axis=0)
        lower_bound = np.percentile(ensemble_array, 5, axis=0)  # 5th percentile
        upper_bound = np.percentile(ensemble_array, 95, axis=0)  # 95th percentile
        
        # Create result lists
        result = []
        confidence_intervals = []
        
        for i, date in enumerate(prediction_dates):
            # Ensure predictions are non-negative
            price = max(0, mean_predictions[i])
            lower = max(0, lower_bound[i])
            upper = max(0, upper_bound[i])
            
            result.append((date, price))
            confidence_intervals.append((date, lower, upper))
        
        return result, confidence_intervals
    
    def predict_multi_step(self, commodity, region, num_weeks=4):
        """
        Predict prices using a multi-step approach (predict one week at a time).
        
        Args:
            commodity (str): Commodity to predict for
            region (str): Region to predict for
            num_weeks (int): Number of weeks to predict
        
        Returns:
            list: List of (date, price) tuples for the predicted weeks
        """
        # Get historical data for the commodity and region
        historical_data = self.data_pipeline.get_data_for_commodity_region(commodity, region)
        
        if historical_data.empty:
            print(f"No historical data available for {commodity} in {region}")
            return []
        
        # Get model
        model, model_type, feature_cols, metrics = self.get_model(commodity, region)
        
        if model is None:
            print(f"No model available for {commodity}")
            return []
        
        # Make a copy of the historical data
        data = historical_data.copy()
        
        # Get the latest date in the data
        latest_date = data['date'].max()
        
        # Initialize result list
        result = []
        
        # Predict one week at a time
        for i in range(num_weeks):
            # Calculate the next date
            next_date = latest_date + timedelta(days=7 * (i + 1))
            
            # Prepare features for the next date
            future_features = self.feature_engineer.prepare_features_for_prediction(
                data, prediction_dates=[next_date]
            )
            
            if future_features.empty:
                print(f"Failed to generate features for prediction at week {i+1}")
                break
            
            # Make prediction
            prediction = self.model_trainer.predict(model, model_type, future_features, feature_cols)
            
            if prediction is None:
                print(f"Failed to generate prediction at week {i+1}")
                break
            
            # Ensure prediction is non-negative
            price = max(0, prediction[0])
            
            # Add to result
            result.append((next_date, price))
            
            # Add the prediction to the data for the next iteration
            new_row = future_features.iloc[0].copy()
            new_row['price'] = price
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        
        return result
