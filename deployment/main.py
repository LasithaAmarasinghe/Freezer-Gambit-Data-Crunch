from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, timedelta
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

app = FastAPI(
    title="AgroChill API",
    description="API for crop price prediction and strategic insights for the Freezer Gambit strategy",
    version="1.0.0"
)

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

class InsightsRequest(BaseModel):
    crop: str
    region: str

class FrozenPriceProjection(BaseModel):
    week: int
    date: str
    fresh_price: float
    frozen_price: float
    depreciation_percentage: float

class InsightsResponse(BaseModel):
    crop: str
    region: str
    crop_type: str
    price_projections: List[FrozenPriceProjection]
    best_selling_week: int
    best_selling_date: str
    best_selling_price: float
    is_freezing_recommended: bool
    recommendation: str
    insights: List[str]

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

@app.post("/api/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict(request: PredictRequest):
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


@app.post("/api/data/weather", tags=["data"])
async def add_weather_data(data: WeatherData):
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

@app.post("/api/data/prices", tags=["data"])
async def add_price_data(data: PriceData):
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

@app.post("/api/insights", response_model=InsightsResponse, tags=["insights"])
async def get_insights(request: InsightsRequest):
    """
    Provides strategic insights for the Freezer Gambit strategy based on price predictions.

    This endpoint analyzes price predictions for a specific crop and region, calculates
    depreciation rates for frozen produce, and provides recommendations on when to sell
    (fresh vs. frozen) to maximize profits.
    """
    try:
        # Get price predictions first
        y_train_scaled, y_scaler, _ = load_data()
        model = load_model()

        # Load original dataframe to find the last date and crop type
        price_df = pd.read_csv("./Datasets/PriceData/train_data.csv")

        # Filter the group for date info and crop type
        group_df = price_df[(price_df['Region'] == request.region) & (price_df['Commodity'] == request.crop)]
        if group_df.empty:
            raise HTTPException(status_code=400, detail="No matching crop or region found")

        # Get crop type
        crop_type = "Unknown"
        if "Type" in group_df.columns:
            crop_type = group_df["Type"].iloc[0]

        # Get last date and generate future dates
        last_date = pd.to_datetime(group_df['Date'].max())
        future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=4, freq='7D')

        # Find the correct series and make prediction
        for i, series in enumerate(y_train_scaled):
            if series.static_covariates['Region'].iloc[0] == request.region and \
               series.static_covariates['Commodity'].iloc[0] == request.crop:

                y_pred = model.predict(n=4, series=y_train_scaled[i])
                y_pred = y_scaler.inverse_transform(y_pred)

                # Extract prediction values
                prediction_values = [round(float(val[0]), 2) for val in y_pred.values()]

                # Calculate depreciation rates for frozen produce
                # Based on problem statement: frozen produce depreciates at a predictable rate each week
                # We'll use a 10% depreciation rate per week as mentioned in the example
                depreciation_rate = 0.10

                # Calculate frozen prices for each week
                # Week 1: Fresh price from week 0 with 10% depreciation
                # Week 2: Fresh price from week 0 with 19% depreciation (compound)
                # Week 3: Fresh price from week 0 with 27.1% depreciation (compound)
                # Week 4: Fresh price from week 0 with 34.39% depreciation (compound)

                # Get current price (before predictions)
                current_price = group_df['Price per Unit (Silver Drachma/kg)'].iloc[-1]

                # Calculate frozen prices and create projections
                price_projections = []
                frozen_prices = []

                for week, (date_val, fresh_price) in enumerate(zip(future_dates, prediction_values)):
                    # Calculate frozen price (current price with compound depreciation)
                    depreciation_percentage = (1 - (1 - depreciation_rate) ** (week + 1)) * 100
                    frozen_price = round(current_price * ((1 - depreciation_rate) ** (week + 1)), 2)
                    frozen_prices.append(frozen_price)

                    price_projections.append({
                        "week": week + 1,
                        "date": str(date_val.date()),
                        "fresh_price": fresh_price,
                        "frozen_price": frozen_price,
                        "depreciation_percentage": round(depreciation_percentage, 2)
                    })

                # Determine best selling strategy
                # Compare fresh prices vs frozen prices for each week
                fresh_prices = prediction_values

                # Find the week with the highest fresh price
                best_fresh_week = fresh_prices.index(max(fresh_prices)) + 1
                best_fresh_price = max(fresh_prices)
                best_fresh_date = str(future_dates[best_fresh_week - 1].date())

                # Determine if freezing is recommended
                # Compare the best fresh price with the frozen price for that week
                frozen_price_at_best_fresh_week = frozen_prices[best_fresh_week - 1]

                # Check if any fresh price in future weeks is better than the frozen price
                is_freezing_recommended = any(fresh_price > frozen_price for fresh_price, frozen_price
                                             in zip(fresh_prices, frozen_prices))

                # Generate recommendation
                if is_freezing_recommended:
                    recommendation = f"Freeze the {request.crop} and sell in week {best_fresh_week} ({best_fresh_date}) for {best_fresh_price} Silver Drachma/kg."
                else:
                    recommendation = f"Sell the {request.crop} fresh immediately as freezing will not increase profits."

                # Generate insights
                insights = []

                # Price trend insight
                if fresh_prices[3] > fresh_prices[0]:
                    insights.append(f"Prices for {request.crop} in {request.region} show an upward trend over the next 4 weeks.")
                elif fresh_prices[3] < fresh_prices[0]:
                    insights.append(f"Prices for {request.crop} in {request.region} show a downward trend over the next 4 weeks.")
                else:
                    insights.append(f"Prices for {request.crop} in {request.region} remain relatively stable over the next 4 weeks.")

                # Freezing benefit insight
                if is_freezing_recommended:
                    price_difference = best_fresh_price - frozen_price_at_best_fresh_week
                    percentage_gain = (price_difference / frozen_price_at_best_fresh_week) * 100
                    insights.append(f"Freezing will yield approximately {round(percentage_gain, 2)}% more profit compared to selling frozen produce.")
                else:
                    insights.append("The depreciation rate of frozen produce outweighs potential future price increases.")

                # Market volatility insight
                price_variance = np.var(fresh_prices)
                if price_variance > 5:
                    insights.append(f"The market for {request.crop} in {request.region} shows high volatility, suggesting careful timing of sales.")
                else:
                    insights.append(f"The market for {request.crop} in {request.region} shows low volatility, suggesting consistent demand.")

                # Construct the response
                response = {
                    "crop": request.crop,
                    "region": request.region,
                    "crop_type": crop_type,
                    "price_projections": price_projections,
                    "best_selling_week": best_fresh_week,
                    "best_selling_date": best_fresh_date,
                    "best_selling_price": best_fresh_price,
                    "is_freezing_recommended": is_freezing_recommended,
                    "recommendation": recommendation,
                    "insights": insights
                }

                return response

        raise HTTPException(status_code=400, detail="No matching crop or region found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
