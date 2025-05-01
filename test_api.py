import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint."""
    response = requests.get(f"{BASE_URL}/")
    print(f"Root endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_status():
    """Test the status endpoint."""
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"Status endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction(commodity, region):
    """Test the prediction endpoint."""
    payload = {
        "commodity": commodity,
        "region": region,
        "num_weeks": 4,
        "ensemble": True
    }
    response = requests.post(f"{BASE_URL}/api/predict", json=payload)
    print(f"Prediction endpoint response for {commodity} in {region}: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print()

def test_add_price_data():
    """Test adding new price data."""
    # Create sample price data
    today = datetime.now()
    payload = [
        {
            "date": (today - timedelta(days=7)).isoformat(),
            "region": "Arcadia",
            "commodity": "Amaranth Leaves",
            "price": 350.0,
            "type": "Vegetable"
        },
        {
            "date": today.isoformat(),
            "region": "Arcadia",
            "commodity": "Amaranth Leaves",
            "price": 355.0,
            "type": "Vegetable"
        }
    ]
    response = requests.post(f"{BASE_URL}/api/data/prices", json=payload)
    print(f"Add price data endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_add_weather_data():
    """Test adding new weather data."""
    # Create sample weather data
    today = datetime.now()
    payload = [
        {
            "date": (today - timedelta(days=7)).isoformat(),
            "region": "Arcadia",
            "temperature": 298.0,
            "rainfall": 10.0,
            "humidity": 75.0,
            "yield_impact": 0.8
        },
        {
            "date": today.isoformat(),
            "region": "Arcadia",
            "temperature": 300.0,
            "rainfall": 5.0,
            "humidity": 70.0,
            "yield_impact": 0.9
        }
    ]
    response = requests.post(f"{BASE_URL}/api/data/weather", json=payload)
    print(f"Add weather data endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_train_model():
    """Test training a model."""
    payload = {
        "commodity": "Amaranth Leaves",
        "region": "Arcadia",
        "model_type": "xgboost",
        "n_trials": 5
    }
    response = requests.post(f"{BASE_URL}/api/train", json=payload)
    print(f"Train model endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_commodities():
    """Test getting commodities list."""
    response = requests.get(f"{BASE_URL}/api/commodities")
    print(f"Commodities endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_regions():
    """Test getting regions list."""
    response = requests.get(f"{BASE_URL}/api/regions")
    print(f"Regions endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing AgroChill Price Forecasting API...")
    
    # Test root endpoint
    test_root()
    
    # Test status endpoint
    test_status()
    
    # Test commodities and regions endpoints
    test_commodities()
    test_regions()
    
    # Test prediction endpoint
    test_prediction("Butternut Squash", "Gotham")
    
    # Test adding new data
    test_add_price_data()
    test_add_weather_data()
    
    # Test training a model
    test_train_model()
    
    print("API testing completed!")
