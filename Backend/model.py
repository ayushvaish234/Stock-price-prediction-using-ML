#Author : Ayush vaish
#Roll no : 230231017

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import download_stock_data,prepare_lstm_data



def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_forecast_lstm(df, forecast_days=1):
    X, y, X_forecast, scaler = prepare_lstm_data(df, forecast_days)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    predictions = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, predictions)
    
    print("MEAN Absolute Error:",mae)
    accuracy = 1 - (mae / max(y_test))
    print("Estimated Accuracy:", f'{accuracy:.2f}')

    forecast_scaled = model.predict(X_forecast)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    current_price = df['Close'].iloc[-1] 

    return forecast, current_price

# if __name__ == "__main__":
#     symbol = sys.argv[1]
#     start = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
#     end = datetime.strptime(sys.argv[3], "%Y-%m-%d").date()
#     forecast_days = int(sys.argv[4])

#     df = download_stock_data(symbol, start, end)
#     forecast, current_price = train_and_forecast_lstm(df, forecast_days)

#     output = {
#         "symbol": symbol,
#         "current_price": round(float(current_price), 2),
#         "forecast": [round(float(p), 2) for p in forecast]
#     }

#     print(json.dumps(output))
