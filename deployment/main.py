from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import pandas as pd
import numpy as np
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import uvicorn
import joblib

app = FastAPI(title="Crop Price Prediction API", version="1.0.0")

# ---------------------- Data Models ----------------------

class PredictRequest(BaseModel):
    crop: str
    region: str

class PredictionResponse(BaseModel):
    crop: str
    region: str
    predictions: List[dict]

class WeatherData(BaseModel):
    date: date
    region: str
    weatherData: dict

class PriceData(BaseModel):
    date: date
    crop: str
    region: str
    priceData: dict

# ---------------------- Training ----------------------
def build_model():
    torch.manual_seed(1)
    np.random.seed(1)

    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=4,
        n_epochs=100,
        random_state=0,
        pl_trainer_kwargs={
            "accelerator": "cpu",  
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    )
    return model

def retrain_model(model_path="./models/nbeats_model"):
    try:
        y_train_scaled, _ ,past_cov_train_scaled = load_data()

        if len(y_train_scaled) == 0:
            raise ValueError("No time series data found to train the model.")

        model = build_model()
        model.fit(y_train_scaled, past_cov_train_scaled)  
        model.save(model_path)
        return {"message": "Model retrained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def retrain_whether_impact_model():
    df = pd.read_csv("./Datasets/WeatherData/train_data.csv")
    df = df.dropna()

    features = df[["Region", "Temperature (K)", "Rainfall (mm)", "Humidity (%)"]]
    target = df["Crop Yield Impact Score"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Region']),
    ], remainder='passthrough')

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    model_pipeline.fit(features, target)
    joblib.dump(model_pipeline, "crop_impact_model.pkl")

def retrain_all_models():
    try:
        retrain_model()
        retrain_whether_impact_model()
        return {"message": "All models retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Utilities ----------------------

def load_data():
    price_path = "./Datasets/PriceData/train_data.csv"
    weather_path = "./Datasets/WeatherData/train_data.csv"
    
    if not os.path.exists(price_path) or not os.path.exists(weather_path):
        raise FileNotFoundError("Required data files are missing.")

    weather_df = pd.read_csv(weather_path)
    price_df = pd.read_csv(price_path)

    price_df['Date'] = pd.to_datetime(price_df['Date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    weather_agg = weather_df.groupby(['Region', 'Date']).mean().reset_index()
    full_df = pd.merge(price_df, weather_agg, on=['Region', 'Date'], how='left')
    full_df = full_df.drop_duplicates(subset=['Region', 'Date', 'Commodity'])
    full_df['RoundedDate'] = full_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)

    agg_df = full_df.groupby(['Region', 'Commodity', 'RoundedDate']).agg(
        {col: 'mean' for col in full_df.select_dtypes(include='number').columns}
    ).reset_index()

    def enforce_weekly_frequency(group):
        full_range = pd.date_range(start=group['RoundedDate'].min(),
                                   end=group['RoundedDate'].max(), freq='7D')
        group = group.set_index('RoundedDate').reindex(full_range)
        group['Region'] = group['Region'].fillna(method='ffill')
        group['Commodity'] = group['Commodity'].fillna(method='ffill')
        group = group.rename_axis('RoundedDate').reset_index()
        return group

    fixed_df = agg_df.groupby(['Region', 'Commodity']).apply(enforce_weekly_frequency).reset_index(drop=True)
    fixed_df = fixed_df.fillna(fixed_df.mean(numeric_only=True))

    y_all = TimeSeries.from_group_dataframe(fixed_df, group_cols=['Region', 'Commodity'],
                                            time_col='RoundedDate',
                                            value_cols=['Price per Unit (Silver Drachma/kg)'],
                                            freq='7D')
    
    past_cov_all = TimeSeries.from_group_dataframe(fixed_df,
                                           group_cols=['Region','Commodity'],
                                           time_col='RoundedDate',
                                           value_cols=['Crop Yield Impact Score'],
                                           freq='7D') 
    

    y_scaler, past_cov_scaler = Scaler(), Scaler()
    y_train_scaled = y_scaler.fit_transform(y_all)
    past_cov_train_scaled = past_cov_scaler.fit_transform(past_cov_all)

    return y_train_scaled, y_scaler, past_cov_train_scaled

def load_model():
    return NBEATSModel.load("./models/nbeats_model")

# ---------------------- Endpoints ----------------------

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    try:
        y_train_scaled, y_scaler, _ = load_data()
        model = load_model()

        # Load original full dataframe to find the last date
        fixed_df = pd.read_csv("./Datasets/PriceData/train_data.csv")
        

        # Filter the group for date info
        group_df = fixed_df[(fixed_df['Region'] == request.region) & (fixed_df['Commodity'] == request.crop)]
        if group_df.empty:
            raise HTTPException(status_code=400, detail="No matching crop or region found")

        last_date = pd.to_datetime(group_df['Date'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=4, freq='7D')

        # Find the correct series and make prediction
        for i, series in enumerate(y_train_scaled):
            if series.static_covariates['Region'].iloc[0] == request.region and \
               series.static_covariates['Commodity'].iloc[0] == request.crop:

                y_pred = model.predict(n=4, series=y_train_scaled[i])
                y_pred = y_scaler.inverse_transform(y_pred)

                response = {
                    "crop": request.crop,
                    "region": request.region,
                    "predictions": [
                        {
                            "prediction_index": idx,
                            "date": str(future_dates[idx].date()),
                            "price": round(float(val[0]), 2),
                        } for idx, val in enumerate(y_pred.values())
                    ]
                }
                return response

        raise HTTPException(status_code=400, detail="No matching crop or region found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/weather")
def add_weather_data(data: WeatherData):
    try:
        model = joblib.load("./models/crop_impact_model.pkl")

        # Extract and validate weather fields
        temp = data.weatherData.get("temp")
        rainfall = data.weatherData.get("rainfall")
        humidity = data.weatherData.get("humidity")

        if None in [temp, rainfall, humidity, data.region]:
            raise HTTPException(status_code=400, detail="Missing weather or region fields")

        # Construct DataFrame with correct columns
        input_data = pd.DataFrame([{
            "Region": data.region,
            "Temperature (K)": temp,
            "Rainfall (mm)": rainfall,
            "Humidity (%)": humidity
        }])

        predicted_score = model.predict(input_data)[0]

        # Create final row
        row = {
            "Date": data.date,
            "Region": data.region,
            "Temperature (K)": temp,
            "Rainfall (mm)": rainfall,
            "Humidity (%)": humidity,
            "Crop Yield Impact Score": round(predicted_score, 2)
        }

        # Append to CSV
        df = pd.DataFrame([row])
        df.to_csv("./Datasets/WeatherData/train_data.csv", mode='a', header=False, index=False)

        return {"message": "Weather data with predicted score stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/prices")
def add_price_data(data: PriceData):
    try:
        existing_df = pd.read_csv("./Datasets/PriceData/train_data.csv")
        crop_rows = existing_df[existing_df["Commodity"] == data.crop]
        if not crop_rows.empty and "Type" in crop_rows.columns:
            crop_type = crop_rows["Type"].iloc[0]
        else:
            crop_type = "Unknown"
        df = pd.DataFrame([{"Date": data.date, "Region": data.region, "Commodity": data.crop,
                            "Price per Unit (Silver Drachma/kg)": data.priceData['price'],"Type": crop_type}])
        df.to_csv("./Datasets/PriceData/train_data.csv", mode='a', header=False, index=False)
        return {"message": "Price data stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
