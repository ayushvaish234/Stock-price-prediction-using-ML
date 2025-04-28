from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
from model_lstm import train_and_forecast_lstm
from model_xgboost import train_and_forecast_xgboost
from utils import delete_existing_file
import yfinance as yf
from preprocess import download_stock_data
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow all origins by default
from utils import delete_existing_file, ensure_directory_exists

def generate_weighted_test_graphs(y_test_dates, y_test, predictions_lstm, predictions_xgboost, weighted_forecast, forecast_lstm, forecast_xgboost, weights, scaler, graphs_dir):
    """Generate graphs for weighted predictions using test data."""
    # Calculate weighted predictions for test data
    weighted_predictions = (
        weights['lstm'] * predictions_lstm + weights['xgboost'] * predictions_xgboost
    )

    # Inverse transform the predictions and actual values
    y_test_flat = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_predictions_flat = scaler.inverse_transform(predictions_lstm.reshape(-1, 1)).flatten()
    xgboost_predictions_flat = scaler.inverse_transform(predictions_xgboost.reshape(-1, 1)).flatten()
    weighted_predictions_flat = scaler.inverse_transform(weighted_predictions.reshape(-1, 1)).flatten()

    print("Y_test_shape:", y_test.shape)
    print("Weighted_predictions_shape:", weighted_predictions.shape)
    print("Y_test_flat_shape:", y_test_flat.shape)
    print("y_test_dates_shape:", len(y_test_dates))

    # Generate Actual vs Predicted graph for Weighted Test Data
    actual_vs_predicted_path = os.path.join(graphs_dir, "actual_vs_predicted_weighted.png")
    delete_existing_file(actual_vs_predicted_path)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_dates, y_test_flat, label="Actual", color="blue")
    plt.plot(y_test_dates, weighted_predictions_flat, label="Weighted Predicted", color="red")
    plt.title("Actual vs Predicted Stock Prices (Weighted Test Data)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(actual_vs_predicted_path)
    plt.close()

    # Forecasted Prices
    forecasted_prices_path = os.path.join(graphs_dir, "forecasted_prices_weighted.png")
    delete_existing_file(forecasted_prices_path)
    plt.figure(figsize=(10, 6))

    # Extract dates and values for each forecast
    dates = [pd.to_datetime(item['date']).date() for item in weighted_forecast]  # Convert to date objects
    weighted_values = [item['value'] for item in weighted_forecast]
    lstm_values = [item['value'] for item in forecast_lstm]
    xgboost_values = [item['value'] for item in forecast_xgboost]

    plt.plot(dates, weighted_values, label="Weighted Forecast", color="green")
    plt.plot(dates, lstm_values, label="LSTM Forecast", color="blue", linestyle="--")
    plt.plot(dates, xgboost_values, label="XGBoost Forecast", color="red", linestyle="--")
    plt.title("Forecasted Stock Prices (Weighted, LSTM, XGBoost)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(forecasted_prices_path)
    plt.close()

    # Generate Residuals Histogram for Weighted Test Data
    residuals_histogram_path = os.path.join(graphs_dir, "residuals_histogram_weighted.png")
    delete_existing_file(residuals_histogram_path)
    plt.figure(figsize=(10, 6))
    residuals = y_test_flat - weighted_predictions_flat
    plt.hist(residuals, bins=20, color="darkorchid", edgecolor="black")
    plt.title("Residuals Histogram (Weighted Test Data)")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(residuals_histogram_path)
    plt.close()

    # Generate Comparison Graph for Actual, LSTM, XGBoost, and Hybrid Predictions
    comparison_graph_path = os.path.join(graphs_dir, "comparison_predictions.png")
    delete_existing_file(comparison_graph_path)
    plt.figure(figsize=(12, 8))
    plt.plot(y_test_dates, y_test_flat, label="Actual", color="blue", linewidth=2)
    plt.plot(y_test_dates, lstm_predictions_flat, label="LSTM Predicted", color="orange", linestyle="--")
    plt.plot(y_test_dates, xgboost_predictions_flat, label="XGBoost Predicted", color="green", linestyle=":")
    plt.plot(y_test_dates, weighted_predictions_flat, label="Hybrid Predicted", color="red", linestyle="-")
    plt.title("Comparison of Predictions (Actual, LSTM, XGBoost, Hybrid)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(comparison_graph_path)
    plt.close()

    return {
        "actual_vs_predicted_weighted": "actual_vs_predicted_weighted.png",
        "forecasted_prices_weighted": "forecasted_prices_weighted.png",
        "residuals_histogram_weighted": "residuals_histogram_weighted.png",
        "comparison_predictions": "comparison_predictions.png"  # New graph path
    }




@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    forecast_days = data.get('forecast_days', 7)  # Default to 7 days if not provided

    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    try:
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years data

        df = download_stock_data(symbol.upper(), start_date, end_date)

        if df.empty:
            return jsonify({'error': 'No data found for this symbol'}), 404

        # Call the LSTM model function
        current_price, forecast_lstm, predictions_lstm, y_test_dates, scaler, graph_paths_lstm = train_and_forecast_lstm(df, forecast_days)

        # Call the XGBoost model function
        _, forecast_xgboost, predictions_xgboost, y_test, graph_paths_xgboost = train_and_forecast_xgboost(df, forecast_days)

        lstm_weight = 0.5
        xgboost_weight = 0.5
        weights = {"lstm": lstm_weight, "xgboost": xgboost_weight}


        # Combine forecast values using a weighted average
        weighted_forecast = []
        for i in range(len(forecast_lstm)):
            date = forecast_lstm[i]['date']
            lstm_value = float(forecast_lstm[i]['value'])
            xgboost_value = float(forecast_xgboost[i]['value'])

            # Calculate the weighted average
            combined_value = (lstm_weight * lstm_value) + (xgboost_weight * xgboost_value)
            weighted_forecast.append({"date": date, "value": round(combined_value, 2)})

        # Debugging output
        print("Final Weighted Forecast:", weighted_forecast)



        # Debugging shapes
        print("predictions_lstm shape:", predictions_lstm.shape)
        print("predictions_xgboost shape:", predictions_xgboost.shape)
        print("y_test shape:", y_test.shape)

        # Use only the first step of predictions_lstm to match the shape of predictions_xgboost and y_test
        predictions_lstm = predictions_lstm[:, 0]  # Take the first step of multi-step predictions
        print("Adjusted predictions_lstm shape:", predictions_lstm.shape)

        # Align shapes if necessary
        min_length = min(len(predictions_lstm), len(predictions_xgboost), len(y_test))
        predictions_lstm = predictions_lstm[:min_length]
        predictions_xgboost = predictions_xgboost[:min_length]
        y_test = y_test[:min_length]
        y_test_dates = y_test_dates[:min_length]

        # Debugging aligned shapes
        print("Aligned predictions_lstm shape:", predictions_lstm.shape)
        print("Aligned predictions_xgboost shape:", predictions_xgboost.shape)
        print("Aligned y_test shape:", y_test.shape)
        print("Aligned y_test_dates length:", len(y_test_dates))

        # Generate graphs for the weighted test data
        graphs_dir = "graphs"
        combined_weighted_graphs = generate_weighted_test_graphs(
            y_test_dates, y_test, predictions_lstm, predictions_xgboost, weighted_forecast, forecast_lstm, forecast_xgboost, weights, scaler=scaler, graphs_dir=graphs_dir
        )

        # Ensure all values in forecasts are JSON serializable
        forecast_lstm = [{"date": item["date"], "value": round(float(item["value"]), 2)} for item in forecast_lstm]
        forecast_xgboost = [{"date": item["date"], "value": round(float(item["value"]), 2)} for item in forecast_xgboost]



        result = {
            'symbol': symbol,
            'current_price': round(float(current_price), 2),  # Return current price only once
            'lstm': {
                "forecast": forecast_lstm,
                'graphs': graph_paths_lstm  # Include LSTM graph paths
            },
            'xgboost': {
                "forecast": forecast_xgboost,
                'graphs': graph_paths_xgboost  # Include XGBoost graph paths
            },
            'hybrid': {
                "forecast": weighted_forecast,  # Include weighted forecast
                "graphs": combined_weighted_graphs,  # Include weighted test graph paths
            }
        }

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500




@app.route('/stock-info', methods=['POST'])
def stock_info():
    data = request.get_json()
    symbol = data.get('symbol')

    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    try:
        # Fetch stock information using yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Extract relevant fields
        stock_info = {
            'name': info.get('longName', 'N/A'),
            'symbol': info.get('symbol', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'about': info.get('longBusinessSummary', 'N/A'),  # About Company
            'peRatio': info.get('trailingPE', 'N/A'),  # P/E Ratio
            'allTimeHigh': info.get('fiftyTwoWeekHigh', 'N/A'),  # All-Time High
            'allTimeLow': info.get('fiftyTwoWeekLow', 'N/A'),  # All-Time Low
        }
        return jsonify(stock_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/graph/<graph_name>', methods=['GET'])
def get_graph(graph_name):
    """
    Endpoint to serve graph images to the client.
    """
    graphs_dir = "graphs"
    graph_path = os.path.join(os.getcwd(), graphs_dir, graph_name)
    print("Graph path:", graph_path)  # Debugging line
    if os.path.exists(graph_path):
        return send_file(graph_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Graph not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Allow external access