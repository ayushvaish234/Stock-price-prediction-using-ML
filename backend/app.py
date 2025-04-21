'''
Author: Sudhanshu Tiwari
Roll no: 230231060
'''

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
from model_lstm import train_and_forecast_lstm
from model_xgboost import train_and_forecast_xgboost
from preprocess import download_stock_data
import os

app = Flask(__name__)
CORS(app)  # Allow all origins by default


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

        df = download_stock_data(symbol, start_date, end_date)

        if df.empty:
            return jsonify({'error': 'No data found for this symbol'}), 404

        # Call the LSTM model function
        current_price, forecast_lstm, graph_paths_lstm = train_and_forecast_lstm(df, forecast_days)

        # Call the XGBoost model function
        _, forecast_xgboost, graph_paths_xgboost = train_and_forecast_xgboost(df, forecast_days)

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