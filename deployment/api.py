from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import uvicorn
import os
import asyncio
import pandas as pd
import numpy as np
import time
import json
import logging
from functools import lru_cache

from .data_pipeline import DataPipeline
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Define paths
PRICE_DATA_PATH = os.path.join("data", "price_data.csv")
WEATHER_DATA_PATH = os.path.join("data", "weather_data.csv")
MODELS_DIR = os.path.join("models")

# Ensure data directory exists
os.makedirs(os.path.dirname(PRICE_DATA_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize components
data_pipeline = DataPipeline(PRICE_DATA_PATH, WEATHER_DATA_PATH)
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer(MODELS_DIR)
predictor = Predictor(data_pipeline, feature_engineer, model_trainer)

# Global variables for retraining
retraining_in_progress = False
last_retrain_time = None
retrain_lock = asyncio.Lock()

# Initialize FastAPI app
app = FastAPI(
    title="AgroChill Price Forecasting API",
    description="Advanced crop price forecasting system for the DataCrunch Final Round",
    version="1.0.0"
)

# Pydantic models for API
class PredictionRequest(BaseModel):
    commodity: str
    region: str
    num_weeks: Optional[int] = 4
    ensemble: Optional[bool] = True

class PredictionItem(BaseModel):
    prediction_index: int
    date: date
    price: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class PredictionResponse(BaseModel):
    commodity: str
    region: str
    predictions: List[PredictionItem]
    model_metrics: Optional[Dict[str, float]] = None

class WeatherData(BaseModel):
    date: datetime
    region: str
    temperature: float = Field(..., description="Temperature in Kelvin")
    rainfall: float = Field(..., description="Rainfall in mm")
    humidity: float = Field(..., description="Humidity percentage")
    yield_impact: float = Field(..., description="Crop Yield Impact Score")

class PriceData(BaseModel):
    date: datetime
    region: str
    commodity: str
    price: float = Field(..., description="Price per unit in Silver Drachma/kg")
    type: Optional[str] = None

class TrainingRequest(BaseModel):
    commodity: Optional[str] = None
    region: Optional[str] = None
    model_type: str = "xgboost"
    n_trials: int = 50

class TrainingResponse(BaseModel):
    status: str
    message: str
    commodity: Optional[str] = None
    region: Optional[str] = None
    model_path: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    commodities: int
    regions: int
    models: int
    retraining_in_progress: bool
    last_retrain_time: Optional[str] = None
    memory_usage_mb: float
    uptime_seconds: float

# Startup event
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AgroChill Price Forecasting API")
    logger.info(f"Loaded {len(data_pipeline.price_df)} price records")
    logger.info(f"Loaded {len(data_pipeline.weather_df)} weather records")

# API endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Get API information and status."""
    return {
        "message": "AgroChill Price Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "uptime": f"{time.time() - start_time:.2f} seconds"
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """
    Predict crop prices for the next 4 weeks for a specific commodity and region.
    """
    try:
        logger.info(f"Prediction request for {request.commodity} in {request.region}")
        
        # Get model metrics
        _, _, _, metrics = predictor.get_model(request.commodity, request.region)
        
        # Make predictions
        if request.ensemble:
            predictions, confidence_intervals = predictor.predict_with_ensemble(
                request.commodity, request.region, num_weeks=request.num_weeks, ensemble_size=5
            )
        else:
            predictions = predictor.predict_prices(
                request.commodity, request.region, num_weeks=request.num_weeks
            )
            confidence_intervals = [(date, price * 0.9, price * 1.1) for date, price in predictions]
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions available for {request.commodity} in {request.region}")
        
        # Format response
        prediction_items = []
        for i, ((date, price), (_, lower, upper)) in enumerate(zip(predictions, confidence_intervals)):
            prediction_items.append(
                PredictionItem(
                    prediction_index=i,
                    date=date.date(),
                    price=round(float(price), 2),
                    lower_bound=round(float(lower), 2),
                    upper_bound=round(float(upper), 2)
                )
            )
        
        return PredictionResponse(
            commodity=request.commodity,
            region=request.region,
            predictions=prediction_items,
            model_metrics=metrics
        )
    
    except Exception as e:
        logger.error(f"Error in predict_prices: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/weather")
async def add_weather_data(data: List[WeatherData], background_tasks: BackgroundTasks):
    """
    Add new weather data to the system.
    """
    try:
        logger.info(f"Adding {len(data)} weather records")
        
        # Convert to list of dictionaries
        weather_records = [record.dict() for record in data]
        
        # Add to data pipeline
        num_added = data_pipeline.add_weather_data(weather_records)
        
        # Schedule retraining if enough new data
        if num_added > 0:
            background_tasks.add_task(schedule_retraining)
        
        return {
            "status": "success",
            "message": f"Added {num_added} weather records",
            "records_added": num_added
        }
    
    except Exception as e:
        logger.error(f"Error in add_weather_data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/prices")
async def add_price_data(data: List[PriceData], background_tasks: BackgroundTasks):
    """
    Add new price data to the system.
    """
    try:
        logger.info(f"Adding {len(data)} price records")
        
        # Convert to list of dictionaries
        price_records = [record.dict() for record in data]
        
        # Add to data pipeline
        num_added = data_pipeline.add_price_data(price_records)
        
        # Schedule retraining if enough new data
        if num_added > 0:
            background_tasks.add_task(schedule_retraining)
        
        return {
            "status": "success",
            "message": f"Added {num_added} price records",
            "records_added": num_added
        }
    
    except Exception as e:
        logger.error(f"Error in add_price_data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a model for a specific commodity and region.
    """
    global retraining_in_progress
    
    if retraining_in_progress:
        return TrainingResponse(
            status="error",
            message="Retraining already in progress",
            commodity=request.commodity,
            region=request.region
        )
    
    try:
        logger.info(f"Training request for {request.commodity} in {request.region}")
        
        # Start training in the background
        background_tasks.add_task(
            train_model_task,
            request.commodity,
            request.region,
            request.model_type,
            request.n_trials
        )
        
        return TrainingResponse(
            status="success",
            message="Training started in the background",
            commodity=request.commodity,
            region=request.region
        )
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """
    Get the status of the system.
    """
    try:
        # Get list of commodities and regions
        commodities = data_pipeline.get_commodity_list()
        regions = data_pipeline.get_region_list()
        
        # Get list of trained models
        models = []
        for commodity in commodities:
            model, model_type, _, _ = model_trainer.load_model(commodity)
            if model is not None:
                models.append({
                    "commodity": commodity,
                    "model_type": model_type
                })
        
        # Get memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        
        return StatusResponse(
            status="running",
            commodities=len(commodities),
            regions=len(regions),
            models=len(models),
            retraining_in_progress=retraining_in_progress,
            last_retrain_time=last_retrain_time,
            memory_usage_mb=memory_usage,
            uptime_seconds=time.time() - start_time
        )
    
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/commodities")
async def get_commodities():
    """
    Get list of available commodities.
    """
    try:
        commodities = data_pipeline.get_commodity_list()
        return {
            "commodities": commodities,
            "count": len(commodities)
        }
    
    except Exception as e:
        logger.error(f"Error in get_commodities: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/regions")
async def get_regions():
    """
    Get list of available regions.
    """
    try:
        regions = data_pipeline.get_region_list()
        return {
            "regions": regions,
            "count": len(regions)
        }
    
    except Exception as e:
        logger.error(f"Error in get_regions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def schedule_retraining():
    """
    Schedule model retraining if needed.
    """
    global retraining_in_progress, last_retrain_time, retrain_lock
    
    # Check if retraining is already in progress
    if retraining_in_progress:
        return
    
    # Check if enough time has passed since last retraining
    if last_retrain_time is not None:
        time_since_last_retrain = (datetime.now() - datetime.fromisoformat(last_retrain_time)).total_seconds()
        if time_since_last_retrain < 3600:  # 1 hour
            return
    
    # Acquire lock
    async with retrain_lock:
        # Double-check that retraining is not already in progress
        if retraining_in_progress:
            return
        
        # Set retraining flag
        retraining_in_progress = True
    
    try:
        logger.info("Starting scheduled retraining")
        
        # Get merged data
        merged_data = data_pipeline.get_merged_data()
        
        # Apply feature engineering
        featured_data = feature_engineer.engineer_features(merged_data)
        
        # Get list of commodities
        commodities = featured_data['commodity'].unique()
        
        # Train models for each commodity
        for commodity in commodities:
            # Filter data for this commodity
            commodity_data = featured_data[featured_data['commodity'] == commodity]
            
            # Prepare training data
            X_train, X_test, y_train, y_test = model_trainer.prepare_training_data(commodity_data)
            
            if X_train is None or X_train.empty:
                logger.warning(f"Insufficient data for {commodity} after preprocessing")
                continue
            
            # Load existing model if available
            model, model_type, feature_cols, _ = model_trainer.load_model(commodity)
            
            if model is not None:
                # Update existing model
                logger.info(f"Updating existing model for {commodity}")
                updated_model = model_trainer.update_model(model, model_type, X_train, y_train)
                
                if updated_model is not None:
                    # Evaluate updated model
                    metrics = model_trainer.evaluate_model(updated_model, model_type, X_test, y_test, feature_cols)
                    
                    # Save updated model
                    model_trainer.save_model(updated_model, commodity, None, model_type, None, feature_cols, metrics)
            else:
                # Train new model
                logger.info(f"Training new model for {commodity}")
                model, params, feature_cols, metrics = model_trainer.train_model_for_commodity(
                    featured_data, commodity, None, 'xgboost', 20
                )
                
                if model is not None:
                    # Save new model
                    model_trainer.save_model(model, commodity, None, 'xgboost', params, feature_cols, metrics)
        
        # Update last retrain time
        last_retrain_time = datetime.now().isoformat()
        logger.info("Scheduled retraining completed")
    
    except Exception as e:
        logger.error(f"Error during retraining: {e}", exc_info=True)
    
    finally:
        # Reset retraining flag
        retraining_in_progress = False

async def train_model_task(commodity, region, model_type, n_trials):
    """
    Train a model for a specific commodity and region.
    """
    global retraining_in_progress, last_retrain_time
    
    # Set retraining flag
    retraining_in_progress = True
    
    try:
        logger.info(f"Starting model training for {commodity} in {region}")
        
        # Get merged data
        merged_data = data_pipeline.get_merged_data()
        
        # Apply feature engineering
        featured_data = feature_engineer.engineer_features(merged_data)
        
        # Train model
        if commodity is not None:
            # Train for specific commodity
            model, params, feature_cols, metrics = model_trainer.train_model_for_commodity(
                featured_data, commodity, region, model_type, n_trials
            )
            
            if model is not None:
                # Save model
                model_trainer.save_model(model, commodity, region, model_type, params, feature_cols, metrics)
                logger.info(f"Model for {commodity} in {region} trained successfully")
        else:
            # Train for all commodities
            logger.info("Training models for all commodities")
            model_trainer.train_all_commodity_models(featured_data, model_type, n_trials, region is not None)
        
        # Update last retrain time
        last_retrain_time = datetime.now().isoformat()
    
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
    
    finally:
        # Reset retraining flag
        retraining_in_progress = False

# Run the application
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
