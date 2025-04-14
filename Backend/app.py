#Author: Sudhanshu Tiwari
#Roll no : 230231060

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from model import train_and_forecast_lstm
from preprocess import download_stock_data

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    forecast_days = data.get('forecast_days', 1)

    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    try:
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years data

        df = download_stock_data(symbol, start_date, end_date)

        if df.empty:
            return jsonify({'error': 'No data found for this symbol'}), 404

        forecast, current_price = train_and_forecast_lstm(df, forecast_days)

        result = {
            'symbol': symbol,
            'forecast': [round(float(val), 2) for val in forecast],  # Convert to float here
            'current_price': round(float(df['Close'].iloc[-1]), 2),
        }


        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
