import pandas as pd
import numpy as np
import sys
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from utils import delete_existing_file, ensure_directory_exists
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import download_stock_data, prepare_data

def build_lstm_model(input_shape, forecast_days):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicitly define input layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(forecast_days))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_forecast_lstm(df, forecast_days):
    """Train the LSTM model and generate forecasts."""
    # Ensure the graphs directory exists
    graphs_dir = "graphs"
    ensure_directory_exists(graphs_dir)

    # Prepare data
    X, y, X_forecast, scaler, y_dates = prepare_data(df, forecast_days)

    # Ensure y has the correct shape for multi-step forecasting
    y = np.array([y[i:i + forecast_days] for i in range(len(y) - forecast_days + 1)])
    y_dates = y_dates[:len(y)]  # Adjust y_dates to match the new length of y
    X = X[:len(y)]  # Adjust X to match the new y length

    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    y_test_dates = y_dates[split:]  # Get the dates corresponding to y_test

    # Debugging shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Build and train the model
    model = build_lstm_model((X.shape[1], 1), forecast_days)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=0)

    # Predictions and metrics
    predictions = model.predict(X_test)

    forecast = []
    input_sequence = X_forecast[0]
    last_date = df.index[-1]  # Get the last date from the dataframe
    print("Last date in the dataframe:", last_date)

    for i in range(forecast_days):
        # Predict the next step
        forecast_scaled = model.predict(input_sequence.reshape(1, -1, 1))
        forecast_value = scaler.inverse_transform(forecast_scaled).flatten()[0]
        forecast_date = last_date + pd.Timedelta(days=i + 1)  # Increment the date

        # Append the forecasted date and value
        forecast.append((forecast_date, forecast_value))
        print(f"Iteration {i}: Forecasted date: {forecast_date}, value: {forecast_value}")

        # Update the input sequence for the next prediction
        input_sequence = np.append(input_sequence[1:], forecast_scaled).reshape(-1, 1)
    

    current_price = float(df['Close'].iloc[-1])

    # Generate graphs
    generate_graphs(y_test_dates, y_test, predictions, forecast, history, scaler, graphs_dir)
    forecast = [{"date": str(date), "value": round(float(value), 2)} for date, value in forecast]


    # Return forecast, current price, and paths to the generated graphs
    return current_price, forecast, predictions, y_test_dates, scaler, {
        "actual_vs_predicted_lstm": "actual_vs_predicted_lstm.png",
        "forecasted_prices_lstm": "forecasted_prices_lstm.png",
        "training_vs_validation_loss_lstm": "training_vs_validation_loss_lstm.png",
        "residuals_histogram_lstm": "residuals_histogram_lstm.png"
    }


def generate_graphs(y_test_dates, y_test, predictions, forecast, history, scaler, graphs_dir):
    """Generate and save graphs for analysis."""
    # Actual vs Predicted
    actual_vs_predicted_path = os.path.join(graphs_dir, "actual_vs_predicted_lstm.png")
    delete_existing_file(actual_vs_predicted_path)
    plt.figure(figsize=(10, 6))

    # Use only the first forecasted step for each sample
    y_test_flat = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()
    predictions_flat = scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()

    # Use y_test_dates for the x-axis
    dates = y_test_dates

    if len(dates) != len(y_test_flat):
        raise ValueError(f"Mismatch between dates ({len(dates)}) and y_test_flat ({len(y_test_flat)})")

    plt.plot(dates, y_test_flat, label="Actual", color="blue")
    plt.plot(dates, predictions_flat, label="Predicted", color="red")
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(actual_vs_predicted_path)
    plt.close()

    # Forecasted Prices
    forecasted_prices_path = os.path.join(graphs_dir, "forecasted_prices_lstm.png")
    delete_existing_file(forecasted_prices_path)
    plt.figure(figsize=(10, 6))

    # Extract dates and values from the forecast tuples
    forecast_dates = [date for date, value in forecast]
    forecast_values = [value for date, value in forecast]

    plt.plot(forecast_dates, forecast_values, label="Forecast", color="green")
    plt.title("Forecasted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(forecasted_prices_path)
    plt.close()

    # Training vs Validation Loss
    training_vs_validation_loss_path = os.path.join(graphs_dir, "training_vs_validation_loss_lstm.png")
    delete_existing_file(training_vs_validation_loss_path)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label="Training Loss", color="orange")
    plt.plot(history.history['val_loss'], label="Validation Loss", color="purple")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(training_vs_validation_loss_path)
    plt.close()

    # Residuals Histogram
    residuals = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)) - scaler.inverse_transform(predictions[:, 0].reshape(-1, 1))
    residuals_histogram_path = os.path.join(graphs_dir, "residuals_histogram_lstm.png")
    delete_existing_file(residuals_histogram_path)
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, color="darkorchid", edgecolor="black")
    plt.title("Residuals (Actual - Predicted)")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.savefig(residuals_histogram_path)
    plt.close()


# if __name__ == "__main__":
#     symbol = sys.argv[1]
#     start = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
#     end = datetime.strptime(sys.argv[3], "%Y-%m-%d").date()
#     forecast_days = int(sys.argv[4])

#     df = download_stock_data(symbol, start, end)
#     current_price, forecast, graph_paths = train_and_forecast_lstm(df, forecast_days)

#     output = {
#         "symbol": symbol,
#         "current_price": round(float(current_price), 2),
#         "forecast": forecast,
#         "graphs": graph_paths
#     }

#     print(json.dumps(output))