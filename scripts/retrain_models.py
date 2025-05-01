import sys
import os

# Add the deployment directory to the path so we can import from main.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deployment'))

# Import the retrain_all_models function from main.py
from main import retrain_all_models

if __name__ == "__main__":
    print("Starting model retraining...")
    result = retrain_all_models()
    print(result["message"])
    print("Model retraining completed.")
