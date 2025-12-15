"""
Utility functions for demand forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def clean_numeric_column(df, column_name):
    """
    Clean and convert a column to numeric type
    
    Args:
        df: DataFrame
        column_name: Name of column to clean
        
    Returns:
        Series with cleaned numeric values
    """
    return pd.to_numeric(df[column_name].str.replace(',', '').str.replace('$', ''), errors='coerce')

def parse_date_column(df, column_name, date_format=None):
    """
    Parse date column to datetime
    
    Args:
        df: DataFrame
        column_name: Name of column to parse
        date_format: Optional date format string
        
    Returns:
        Series with datetime values
    """
    if date_format:
        return pd.to_datetime(df[column_name], format=date_format, errors='coerce')
    else:
        return pd.to_datetime(df[column_name], errors='coerce')

def calculate_moving_average(series, window=3):
    """
    Calculate moving average for a time series
    
    Args:
        series: Pandas Series
        window: Window size for moving average
        
    Returns:
        Series with moving average
    """
    return series.rolling(window=window, min_periods=1).mean()

def calculate_growth_rate(series):
    """
    Calculate period-over-period growth rate
    
    Args:
        series: Pandas Series
        
    Returns:
        Series with growth rates
    """
    return series.pct_change() * 100

def detect_outliers_iqr(df, column_name, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Args:
        df: DataFrame
        column_name: Column to check for outliers
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Boolean mask of outliers
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    return (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

def aggregate_by_period(df, date_column, value_column, period='M'):
    """
    Aggregate data by time period
    
    Args:
        df: DataFrame
        date_column: Name of date column
        value_column: Name of value column to aggregate
        period: Period for aggregation ('D'=day, 'W'=week, 'M'=month, 'Q'=quarter, 'Y'=year)
        
    Returns:
        DataFrame with aggregated data
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy.set_index(date_column, inplace=True)
    
    return df_copy[value_column].resample(period).sum().reset_index()

def create_forecast_dates(start_date, periods, freq='M'):
    """
    Create future date range for forecasting
    
    Args:
        start_date: Starting date
        periods: Number of periods to forecast
        freq: Frequency ('D'=day, 'W'=week, 'M'=month, 'Q'=quarter)
        
    Returns:
        DatetimeIndex with future dates
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)

def calculate_forecast_accuracy(actual, predicted):
    """
    Calculate forecast accuracy metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with accuracy metrics
    """
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse
    }

def prepare_forecast_features(df, date_column, value_column):
    """
    Prepare features for forecasting (trends, seasonality, etc.)
    
    Args:
        df: DataFrame with time series data
        date_column: Name of date column
        value_column: Name of value column
        
    Returns:
        DataFrame with additional features
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Add time-based features
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month
    df_copy['quarter'] = df_copy[date_column].dt.quarter
    df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
    df_copy['week_of_year'] = df_copy[date_column].dt.isocalendar().week
    
    # Add lag features
    df_copy['lag_1'] = df_copy[value_column].shift(1)
    df_copy['lag_3'] = df_copy[value_column].shift(3)
    df_copy['lag_6'] = df_copy[value_column].shift(6)
    
    # Add rolling statistics
    df_copy['rolling_mean_3'] = calculate_moving_average(df_copy[value_column], window=3)
    df_copy['rolling_mean_6'] = calculate_moving_average(df_copy[value_column], window=6)
    
    return df_copy

def simple_exponential_smoothing(series, alpha=0.3):
    """
    Simple exponential smoothing for forecasting
    
    Args:
        series: Time series data
        alpha: Smoothing parameter (0-1)
        
    Returns:
        Smoothed series
    """
    result = [series.iloc[0]]  # Start with first value
    
    for i in range(1, len(series)):
        result.append(alpha * series.iloc[i] + (1 - alpha) * result[i-1])
    
    return pd.Series(result, index=series.index)
