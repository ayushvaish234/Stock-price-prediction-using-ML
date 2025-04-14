#Author : Anil kachhap 
#Roll no : 230231011

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def download_stock_data(symbol: str, start_date, end_date):
    # Download historical stock data from Yahoo Finance
    symbol = symbol.upper()
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return df

def prepare_lstm_data(df: pd.DataFrame, forecast_days: int, window_size=60):
    # Scale the 'Close' price and prepare the data for LSTM
    df = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data) - forecast_days):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i + forecast_days - 1, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_forecast = scaled_data[-window_size:].reshape(1, window_size, 1)

    return X, y, X_forecast, scaler
