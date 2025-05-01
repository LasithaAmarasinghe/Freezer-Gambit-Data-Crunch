# AgroChill Price Forecasting System

## Overview

This project implements an advanced crop price forecasting system for the DataCrunch Final Round competition ("Legacy of the Market King: The Freezer Gambit"). The system predicts weekly fresh crop prices four weeks ahead across various economic centers, leveraging historical price and weather data.

## Key Features

- **Advanced Modeling**: Uses XGBoost with Optuna hyperparameter tuning
- **Sophisticated Feature Engineering**: Time-based features, lag features, rolling statistics, weather trends, Fourier features, and cross-commodity correlations
- **Ensemble Predictions**: Provides confidence intervals for better decision-making
- **Incremental Learning**: Models can be updated with new data without full retraining
- **API-First Design**: RESTful API for predictions and data ingestion
- **Containerized Deployment**: Docker support for easy deployment

## Project Structure

```
/
├── deployment/                # Core application code
│   ├── __init__.py            # Package initialization
│   ├── data_pipeline.py       # Data loading and preprocessing
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── model_trainer.py       # Model training and hyperparameter tuning
│   ├── predictor.py           # Prediction logic
│   └── api.py                 # FastAPI application
├── data/                      # Data directory (created at runtime)
│   ├── price_data.csv         # Price dataset
│   └── weather_data.csv       # Weather dataset
├── models/                    # Trained models (created at runtime)
├── run.py                     # Application entry point
├── train_models.py            # Script to train all models
├── test_api.py                # Script to test the API
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## API Endpoints

- `POST /api/predict`: Predict prices for a specific commodity and region
- `POST /api/data/weather`: Add new weather data
- `POST /api/data/prices`: Add new price data
- `POST /api/train`: Train a model for a specific commodity
- `GET /api/status`: Get system status
- `GET /api/commodities`: Get list of available commodities
- `GET /api/regions`: Get list of available regions

## Setup and Running

### Prerequisites

- Python 3.10+
- Docker (optional)

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train models:
   ```
   python train_models.py
   ```
4. Run the application:
   ```
   python run.py
   ```
5. Access the API at http://localhost:8000

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t agrochill-forecast:v1.0 .
   ```
2. Run the container:
   ```
   docker run -d -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models --name agrochill-forecast agrochill-forecast:v1.0
   ```
3. Access the API at http://localhost:8000

## Model Training

The system automatically trains models for each commodity when data is available. You can also trigger training manually using the `/api/train` endpoint or by running the `train_models.py` script.

## Performance

- **Accuracy**: Advanced feature engineering and ensemble techniques provide state-of-the-art prediction accuracy
- **Efficiency**: Optimized for low resource usage (under 2GB RAM)
- **Scalability**: Separate models for each commodity allow for parallel processing

## Example Usage

### Making Predictions

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "commodity": "Butternut Squash",
        "region": "Gotham",
        "num_weeks": 4,
        "ensemble": True
    }
)

predictions = response.json()
print(predictions)
```

### Adding New Data

```python
import requests
from datetime import datetime

# Add new price data
response = requests.post(
    "http://localhost:8000/api/data/prices",
    json=[
        {
            "date": datetime.now().isoformat(),
            "region": "Arcadia",
            "commodity": "Amaranth Leaves",
            "price": 350.0,
            "type": "Vegetable"
        }
    ]
)
```

## License

This project is for educational and competition use only.
