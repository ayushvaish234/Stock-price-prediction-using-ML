from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
from model_lstm import train_and_forecast_lstm
from model_xgboost import train_and_forecast_xgboost
from preprocess import download_stock_data
import os

app = Flask(__name__)
CORS(app)  # Allow all origins by default

import yfinance as yf

import yfinance as yf

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    forecast_days = data.get('forecast_days', 7)  # Default to 7 days if not provided

    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    try:
        end_date = datetime.today().date()
        print("End date:", end_date)  # Debugging line
        start_date = end_date - timedelta(days=365 * 2)  # 2 years data

        df = download_stock_data(symbol.upper(), start_date, end_date)


        if df.empty:
            return jsonify({'error': 'No data found for this symbol'}), 404

        # Call the LSTM model function
        current_price, forecast_lstm, graph_paths_lstm = train_and_forecast_lstm(df, forecast_days)

        # Call the XGBoost model function
        _, forecast_xgboost, graph_paths_xgboost = train_and_forecast_xgboost(df, forecast_days)

        lstm_weight = 0.7
        xgboost_weight = 0.3

          # Combine forecast values using a weighted average
        weighted_forecast = []
        for i in range(len(forecast_lstm)):
            date = forecast_lstm[i]['date']
            lstm_value = float(forecast_lstm[i]['value'])
            xgboost_value = float(forecast_xgboost[i]['value'])

            # Calculate the weighted average
            combined_value = (lstm_weight * lstm_value) + (xgboost_weight * xgboost_value)
            weighted_forecast.append({"date": date, "value": round(combined_value, 2)})
        
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
            'combined_forecast': {
                "forecast": weighted_forecast,
                'columns': ['Date', 'Predicted Price']  # Added column headers for weighted forecast
            }
        }

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
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