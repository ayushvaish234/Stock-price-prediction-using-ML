import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def download_stock_data(symbol: str, start_date, end_date):
    # Download historical stock data from Yahoo Finance
    symbol = symbol.upper()
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return df

def prepare_data(df: pd.DataFrame, forecast_days: int = 7, window_size=60):
    # Scale the 'Close' price and prepare the data for LSTM
    df = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y, y_dates = [], [], []
    for i in range(window_size, len(scaled_data) - forecast_days):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i + forecast_days - 1, 0])
        y_dates.append(df.index[i + forecast_days - 1])  # Capture the corresponding date for y

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_forecast = scaled_data[-window_size:].reshape(1, window_size, 1)

    # Debugging: Print X, y, y_dates, and df
    print("Dataframe (df):")
    print(df)
    print("Length of df:", len(df))
    print("\nX shape:", X.shape)
    print("y shape:", y.shape)
    print("y_dates length:", len(y_dates))
    print("y_dates:", y_dates)

    return X, y, X_forecast, scaler, y_dates