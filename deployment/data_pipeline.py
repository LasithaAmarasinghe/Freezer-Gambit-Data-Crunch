import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

class DataPipeline:
    """
    Advanced data pipeline for loading, preprocessing, and managing price and weather data.
    Combines approaches from multiple solutions for optimal data handling.
    """
    
    def __init__(self, price_data_path, weather_data_path):
        """
        Initialize the data pipeline with paths to price and weather data.
        
        Args:
            price_data_path (str): Path to the price dataset CSV
            weather_data_path (str): Path to the weather dataset CSV
        """
        self.price_data_path = price_data_path
        self.weather_data_path = weather_data_path
        
        # Initialize dataframes
        self.price_df = None
        self.weather_df = None
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load and preprocess both price and weather datasets."""
        # Load price data
        self.price_df = pd.read_csv(self.price_data_path)
        
        # Load weather data
        self.weather_df = pd.read_csv(self.weather_data_path)
        
        # Standardize column names (from Solution 5)
        self.price_df.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Commodity': 'commodity',
            'Price per Unit (Silver Drachma/kg)': 'price',
            'Type': 'type'
        }, inplace=True)
        
        self.weather_df.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Temperature (K)': 'temperature',
            'Rainfall (mm)': 'rainfall',
            'Humidity (%)': 'humidity',
            'Crop Yield Impact Score': 'yield_impact'
        }, inplace=True)
        
        # Convert date columns to datetime
        self.price_df['date'] = pd.to_datetime(self.price_df['date'])
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        
        # Sort data for time series consistency
        self.price_df.sort_values(['commodity', 'region', 'date'], inplace=True)
        self.weather_df.sort_values(['region', 'date'], inplace=True)
        
        # Convert categorical columns to category dtype for efficiency
        self.price_df['region'] = self.price_df['region'].astype('category')
        self.price_df['commodity'] = self.price_df['commodity'].astype('category')
        self.price_df['type'] = self.price_df['type'].astype('category')
        self.weather_df['region'] = self.weather_df['region'].astype('category')
        
        print(f"Loaded price data: {self.price_df.shape[0]} rows")
        print(f"Loaded weather data: {self.weather_df.shape[0]} rows")
    
    def get_merged_data(self):
        """
        Merge price and weather data on date and region.
        
        Returns:
            pandas.DataFrame: Merged dataframe with price and weather data
        """
        if self.price_df is None or self.weather_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Merge on date and region
        merged_df = pd.merge(self.price_df, self.weather_df, on=['date', 'region'], how='left')
        
        # Check for missing weather data
        missing_weather = merged_df['temperature'].isnull().sum()
        if missing_weather > 0:
            print(f"Warning: {missing_weather} rows have missing weather data after merge.")
            # Impute missing weather data by region
            for col in ['temperature', 'rainfall', 'humidity', 'yield_impact']:
                merged_df[col] = merged_df.groupby('region')[col].transform(lambda x: x.fillna(x.mean()))
        
        return merged_df
    
    def add_price_data(self, new_data):
        """
        Add new price data to the existing dataset.
        
        Args:
            new_data (list or DataFrame): New price records
        
        Returns:
            int: Number of records added
        """
        if not new_data:
            return 0
        
        # Convert to DataFrame if list
        df_new = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data
        
        # Standardize column names
        df_new.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Commodity': 'commodity',
            'Price per Unit (Silver Drachma/kg)': 'price',
            'Type': 'type'
        }, inplace=True)
        
        # Ensure date is in datetime format
        df_new['date'] = pd.to_datetime(df_new['date'])
        
        # Convert categorical columns to category dtype
        df_new['region'] = df_new['region'].astype('category')
        df_new['commodity'] = df_new['commodity'].astype('category')
        if 'type' in df_new.columns:
            df_new['type'] = df_new['type'].astype('category')
        
        # Append to existing data
        self.price_df = pd.concat([self.price_df, df_new], ignore_index=True)
        
        # Sort for consistency
        self.price_df.sort_values(['commodity', 'region', 'date'], inplace=True, ignore_index=True)
        
        # Save updated data
        self.price_df.to_csv(self.price_data_path, index=False)
        
        return len(df_new)
    
    def add_weather_data(self, new_data):
        """
        Add new weather data to the existing dataset.
        
        Args:
            new_data (list or DataFrame): New weather records
        
        Returns:
            int: Number of records added
        """
        if not new_data:
            return 0
        
        # Convert to DataFrame if list
        df_new = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data
        
        # Standardize column names
        df_new.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Temperature (K)': 'temperature',
            'Rainfall (mm)': 'rainfall',
            'Humidity (%)': 'humidity',
            'Crop Yield Impact Score': 'yield_impact'
        }, inplace=True)
        
        # Ensure date is in datetime format
        df_new['date'] = pd.to_datetime(df_new['date'])
        
        # Convert categorical columns to category dtype
        df_new['region'] = df_new['region'].astype('category')
        
        # Append to existing data
        self.weather_df = pd.concat([self.weather_df, df_new], ignore_index=True)
        
        # Sort for consistency
        self.weather_df.sort_values(['region', 'date'], inplace=True, ignore_index=True)
        
        # Save updated data
        self.weather_df.to_csv(self.weather_data_path, index=False)
        
        return len(df_new)
    
    def get_latest_date(self, commodity=None, region=None):
        """
        Get the latest date in the price dataset for a specific commodity and region.
        
        Args:
            commodity (str, optional): Filter by commodity
            region (str, optional): Filter by region
        
        Returns:
            datetime: Latest date in the dataset
        """
        df = self.price_df.copy()
        
        if commodity:
            df = df[df['commodity'] == commodity]
        if region:
            df = df[df['region'] == region]
        
        if df.empty:
            return None
        
        return df['date'].max()
    
    def get_latest_price(self, commodity, region, n=1):
        """
        Get the latest n price records for a specific commodity and region.
        
        Args:
            commodity (str): Commodity name
            region (str): Region name
            n (int): Number of records to return
        
        Returns:
            pandas.DataFrame: Latest n price records
        """
        subset = self.price_df[(self.price_df['commodity'] == commodity) & 
                              (self.price_df['region'] == region)]
        if subset.empty:
            return None
        
        return subset.sort_values('date').tail(n)
    
    def get_latest_weather(self, region, n=1):
        """
        Get the latest n weather records for a specific region.
        
        Args:
            region (str): Region name
            n (int): Number of records to return
        
        Returns:
            pandas.DataFrame: Latest n weather records
        """
        subset = self.weather_df[self.weather_df['region'] == region]
        if subset.empty:
            return None
        
        return subset.sort_values('date').tail(n)
    
    def get_commodity_list(self):
        """Get list of unique commodities in the dataset."""
        return self.price_df['commodity'].unique().tolist()
    
    def get_region_list(self):
        """Get list of unique regions in the dataset."""
        return self.price_df['region'].unique().tolist()
    
    def get_data_for_commodity_region(self, commodity, region):
        """
        Get merged data for a specific commodity and region.
        
        Args:
            commodity (str): Commodity name
            region (str): Region name
        
        Returns:
            pandas.DataFrame: Filtered and merged dataframe
        """
        merged_df = self.get_merged_data()
        return merged_df[(merged_df['commodity'] == commodity) & 
                         (merged_df['region'] == region)]
