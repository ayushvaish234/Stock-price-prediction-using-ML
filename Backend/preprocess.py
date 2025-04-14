#Author : Anil kachhap 
#Roll no : 230231011

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def download_stock_data(symbol: str, start_date, end_date):
    # Download historical stock data from Yahoo Finance
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return df

def prepare_dataset(df: pd.DataFrame, forecast_days: int):
    # Use only the 'Close' column
    df = df[['Close']].copy()
    # Shift closing prices to create the target
    df['Target'] = df['Close'].shift(-forecast_days)
    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Features and target
    X = df[['Close']].values
    y = df['Target'].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Data for forecasting
    X_forecast = X_scaled[-forecast_days:]
    X_scaled = X_scaled[:-forecast_days]
    y = y[:-forecast_days]

    return X_scaled, y, X_forecast, scaler
