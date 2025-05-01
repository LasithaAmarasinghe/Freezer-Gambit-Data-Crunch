import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

class FeatureEngineer:
    """
    Advanced feature engineering for price prediction models.
    Combines the best techniques from multiple solutions.
    """
    
    @staticmethod
    def create_time_features(df):
        """
        Create time-based features from the date column.
        
        Args:
            df (pandas.DataFrame): Input dataframe with a date column
        
        Returns:
            pandas.DataFrame: Dataframe with additional time features
        """
        df_copy = df.copy()
        
        # Basic time features
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['week_of_year'] = df_copy['date'].dt.isocalendar().week.astype(int)
        df_copy['day_of_week'] = df_copy['date'].dt.dayofweek  # Monday=0, Sunday=6
        df_copy['day_of_year'] = df_copy['date'].dt.dayofyear
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        
        # Quarter and semester for longer-term seasonality
        df_copy['quarter'] = df_copy['date'].dt.quarter
        df_copy['semester'] = ((df_copy['month'] - 1) // 6 + 1).astype(int)
        
        # Cyclical encoding of time features to capture periodicity
        # This helps the model understand that Dec (12) is close to Jan (1)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['week_of_year'] / 52)
        df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['week_of_year'] / 52)
        df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        return df_copy
    
    @staticmethod
    def create_lag_features(df, group_cols, target_col, lags):
        """
        Create lag features for the target column.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            group_cols (list): Columns to group by before creating lags
            target_col (str): Target column to create lags for
            lags (list): List of lag periods to create
        
        Returns:
            pandas.DataFrame: Dataframe with additional lag features
        """
        df_copy = df.sort_values(by=group_cols + ['date']).copy()
        
        # Suppress FutureWarning for groupby observed parameter
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            for lag in lags:
                lag_col_name = f'{target_col}_lag_{lag}'
                df_copy[lag_col_name] = df_copy.groupby(group_cols, observed=False)[target_col].shift(lag)
        
        return df_copy
    
    @staticmethod
    def create_rolling_window_features(df, group_cols, target_col, windows, stats):
        """
        Create rolling window features for the target column.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            group_cols (list): Columns to group by before creating rolling features
            target_col (str): Target column to create rolling features for
            windows (list): List of window sizes
            stats (list): List of statistics to compute (e.g., 'mean', 'std')
        
        Returns:
            pandas.DataFrame: Dataframe with additional rolling window features
        """
        df_copy = df.sort_values(by=group_cols + ['date']).copy()
        
        # Suppress FutureWarning for groupby observed parameter
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            # Group by the specified columns
            grouped = df_copy.groupby(group_cols, observed=False)[target_col]
            
            for window in windows:
                # Shift by 1 to avoid data leakage (use only past data)
                shifted_group = grouped.shift(1)
                
                for stat in stats:
                    roll_col_name = f'{target_col}_roll_{stat}_{window}'
                    
                    # Calculate rolling statistic
                    if stat == 'mean':
                        df_copy[roll_col_name] = shifted_group.rolling(
                            window=window, min_periods=max(1, window // 2)).mean()
                    elif stat == 'std':
                        df_copy[roll_col_name] = shifted_group.rolling(
                            window=window, min_periods=max(1, window // 2)).std()
                    elif stat == 'min':
                        df_copy[roll_col_name] = shifted_group.rolling(
                            window=window, min_periods=max(1, window // 2)).min()
                    elif stat == 'max':
                        df_copy[roll_col_name] = shifted_group.rolling(
                            window=window, min_periods=max(1, window // 2)).max()
                    elif stat == 'median':
                        df_copy[roll_col_name] = shifted_group.rolling(
                            window=window, min_periods=max(1, window // 2)).median()
        
        return df_copy
    
    @staticmethod
    def create_price_momentum_features(df, group_cols, target_col, windows):
        """
        Create price momentum features (percent change over different windows).
        
        Args:
            df (pandas.DataFrame): Input dataframe
            group_cols (list): Columns to group by before creating features
            target_col (str): Target column to create features for
            windows (list): List of window sizes
        
        Returns:
            pandas.DataFrame: Dataframe with additional momentum features
        """
        df_copy = df.sort_values(by=group_cols + ['date']).copy()
        
        # Suppress FutureWarning for groupby observed parameter
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            for window in windows:
                # Calculate percent change over the window
                pct_change_col = f'{target_col}_pct_change_{window}'
                df_copy[pct_change_col] = df_copy.groupby(group_cols, observed=False)[target_col].pct_change(periods=window)
                
                # Calculate acceleration (change in percent change)
                if window > 1:
                    accel_col = f'{target_col}_accel_{window}'
                    df_copy[accel_col] = df_copy.groupby(group_cols, observed=False)[pct_change_col].diff()
        
        return df_copy
    
    @staticmethod
    def create_weather_trend_features(df, weather_cols, windows):
        """
        Create weather trend features (changes in weather metrics).
        
        Args:
            df (pandas.DataFrame): Input dataframe
            weather_cols (list): Weather columns to create trends for
            windows (list): List of window sizes
        
        Returns:
            pandas.DataFrame: Dataframe with additional weather trend features
        """
        df_copy = df.sort_values(by=['region', 'date']).copy()
        
        # Suppress FutureWarning for groupby observed parameter
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            for col in weather_cols:
                for window in windows:
                    # Calculate change over the window
                    change_col = f'{col}_change_{window}'
                    df_copy[change_col] = df_copy.groupby('region', observed=False)[col].diff(periods=window)
                    
                    # Calculate rolling mean to capture trends
                    trend_col = f'{col}_trend_{window}'
                    df_copy[trend_col] = df_copy.groupby('region', observed=False)[col].shift(1).rolling(
                        window=window, min_periods=max(1, window // 2)).mean()
        
        return df_copy
    
    @staticmethod
    def create_cross_commodity_features(df, target_col, n_correlations=5):
        """
        Create cross-commodity correlation features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            target_col (str): Target column to create correlations for
            n_correlations (int): Number of top correlated commodities to use
        
        Returns:
            pandas.DataFrame: Dataframe with additional cross-commodity features
        """
        df_copy = df.copy()
        
        # Pivot to get prices by commodity
        pivot_df = df_copy.pivot_table(
            index=['date', 'region'], 
            columns='commodity', 
            values=target_col
        ).reset_index()
        
        # Merge back to original dataframe
        df_copy = pd.merge(df_copy, pivot_df, on=['date', 'region'], how='left')
        
        # For each commodity, find the most correlated other commodities
        commodities = df_copy['commodity'].unique()
        
        for commodity in commodities:
            # Get data for this commodity
            commodity_data = df_copy[df_copy['commodity'] == commodity]
            
            # Calculate correlations with other commodities
            corr_dict = {}
            for other_commodity in commodities:
                if other_commodity != commodity and other_commodity in df_copy.columns:
                    # Calculate correlation between this commodity's price and the other commodity
                    corr = commodity_data[target_col].corr(commodity_data[other_commodity])
                    if not np.isnan(corr):
                        corr_dict[other_commodity] = abs(corr)  # Use absolute correlation
            
            # Get top correlated commodities
            top_correlated = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)[:n_correlations]
            
            # Create features for top correlated commodities
            for other_commodity, corr in top_correlated:
                df_copy.loc[df_copy['commodity'] == commodity, f'corr_{other_commodity}'] = df_copy[other_commodity]
        
        # Drop the temporary commodity columns
        df_copy = df_copy.drop(columns=commodities)
        
        return df_copy
    
    @staticmethod
    def create_fourier_features(df, col, period, harmonics=3):
        """
        Create Fourier features to capture seasonality.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            col (str): Column to create Fourier features for (e.g., 'day_of_year')
            period (int): Period of the seasonality
            harmonics (int): Number of harmonics to use
        
        Returns:
            pandas.DataFrame: Dataframe with additional Fourier features
        """
        df_copy = df.copy()
        
        for i in range(1, harmonics + 1):
            df_copy[f'{col}_sin_{i}'] = np.sin(2 * np.pi * i * df_copy[col] / period)
            df_copy[f'{col}_cos_{i}'] = np.cos(2 * np.pi * i * df_copy[col] / period)
        
        return df_copy
    
    @staticmethod
    def create_volatility_features(df, group_cols, target_col, windows):
        """
        Create price volatility features.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            group_cols (list): Columns to group by before creating features
            target_col (str): Target column to create features for
            windows (list): List of window sizes
        
        Returns:
            pandas.DataFrame: Dataframe with additional volatility features
        """
        df_copy = df.sort_values(by=group_cols + ['date']).copy()
        
        # Suppress FutureWarning for groupby observed parameter
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            # Calculate returns (percent change)
            returns_col = f'{target_col}_returns'
            df_copy[returns_col] = df_copy.groupby(group_cols, observed=False)[target_col].pct_change()
            
            for window in windows:
                # Calculate rolling volatility (standard deviation of returns)
                vol_col = f'{target_col}_volatility_{window}'
                df_copy[vol_col] = df_copy.groupby(group_cols, observed=False)[returns_col].shift(1).rolling(
                    window=window, min_periods=max(1, window // 2)).std()
                
                # Calculate rolling range (high-low)
                range_col = f'{target_col}_range_{window}'
                rolling_max = df_copy.groupby(group_cols, observed=False)[target_col].shift(1).rolling(
                    window=window, min_periods=max(1, window // 2)).max()
                rolling_min = df_copy.groupby(group_cols, observed=False)[target_col].shift(1).rolling(
                    window=window, min_periods=max(1, window // 2)).min()
                df_copy[range_col] = rolling_max - rolling_min
        
        return df_copy
    
    def engineer_features(self, df, target_col='price'):
        """
        Apply all feature engineering steps to the input dataframe.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            target_col (str): Target column for prediction
        
        Returns:
            pandas.DataFrame: Dataframe with all engineered features
        """
        if df.empty:
            return df
        
        # Define grouping columns used repeatedly
        group_cols = ['region', 'commodity']
        
        # 1. Create time-based features
        df_featured = self.create_time_features(df)
        
        # 2. Create lag features for Price (7, 14, 21, 28, 35, 42 days)
        df_featured = self.create_lag_features(
            df_featured, group_cols, target_col, [7, 14, 21, 28, 35, 42]
        )
        
        # 3. Create rolling window features for Price
        df_featured = self.create_rolling_window_features(
            df_featured, group_cols, target_col, [7, 14, 28], ['mean', 'std', 'min', 'max']
        )
        
        # 4. Create price momentum features
        df_featured = self.create_price_momentum_features(
            df_featured, group_cols, target_col, [7, 14, 28]
        )
        
        # 5. Create weather trend features
        weather_cols = ['temperature', 'rainfall', 'humidity', 'yield_impact']
        df_featured = self.create_weather_trend_features(
            df_featured, weather_cols, [7, 14, 28]
        )
        
        # 6. Create volatility features
        df_featured = self.create_volatility_features(
            df_featured, group_cols, target_col, [7, 14, 28]
        )
        
        # 7. Create Fourier features for day of year (annual seasonality)
        df_featured = self.create_fourier_features(df_featured, 'day_of_year', 365, harmonics=3)
        
        # 8. Create Fourier features for week of year (annual seasonality)
        df_featured = self.create_fourier_features(df_featured, 'week_of_year', 52, harmonics=2)
        
        # 9. Create cross-commodity features (if enough data)
        if len(df['commodity'].unique()) > 5:
            try:
                df_featured = self.create_cross_commodity_features(df_featured, target_col, n_correlations=3)
            except Exception as e:
                print(f"Warning: Could not create cross-commodity features: {e}")
        
        return df_featured
    
    def prepare_features_for_prediction(self, df, target_col='price', prediction_dates=None):
        """
        Prepare features for prediction, including future dates.
        
        Args:
            df (pandas.DataFrame): Historical data
            target_col (str): Target column for prediction
            prediction_dates (list): List of dates to predict for
        
        Returns:
            pandas.DataFrame: Dataframe with features for prediction dates
        """
        if df.empty:
            return df
        
        # Get the latest date in the data
        latest_date = df['date'].max()
        
        # If prediction_dates not provided, generate 4 weeks of future dates
        if prediction_dates is None:
            prediction_dates = [
                latest_date + timedelta(days=7 * (i + 1))
                for i in range(4)
            ]
        
        # Create a dataframe for future dates
        future_records = []
        
        # Get the most recent record to use as a template
        latest_record = df[df['date'] == latest_date].iloc[0].to_dict()
        
        # Create a record for each future date
        for future_date in prediction_dates:
            future_record = latest_record.copy()
            future_record['date'] = future_date
            # Set the target to NaN for future dates
            future_record[target_col] = np.nan
            future_records.append(future_record)
        
        # Create a dataframe with future records
        future_df = pd.DataFrame(future_records)
        
        # Combine historical and future data
        combined_df = pd.concat([df, future_df], ignore_index=True)
        
        # Apply feature engineering to the combined data
        featured_df = self.engineer_features(combined_df, target_col)
        
        # Return only the future dates with their features
        return featured_df[featured_df['date'].isin(prediction_dates)]
