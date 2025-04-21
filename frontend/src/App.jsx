/*
Author : Aman Kumar Maourya
Roll No: 230231008
*/

import React, { useState } from 'react';

function App() {
  const [symbol, setSymbol] = useState('');
  const [forecastDays, setForecastDays] = useState(7); // State for forecast days
  const [currentPrice, setCurrentPrice] = useState('');
  const [forecastLSTM, setForecastLSTM] = useState([]); // Forecast by LSTM
  const [forecastXGBoost, setForecastXGBoost] = useState([]); // Forecast by XGBoost
  const [graphsLSTM, setGraphsLSTM] = useState({}); // Graphs for LSTM
  const [graphsXGBoost, setGraphsXGBoost] = useState({}); // Graphs for XGBoost
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(''); // State to store error messages

  document.title = 'Stock Price Prediction';

  const handlePredict = async () => {
    setLoading(true);
    setError(''); // Clear any previous error
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, forecast_days: forecastDays }), // Send forecastDays to the server
      });
      const data = await response.json();

      if (data.error) {
        setError(data.error); // Set the error message
        setCurrentPrice(''); // Clear current price
        setForecastLSTM([]); // Clear LSTM forecast
        setForecastXGBoost([]); // Clear XGBoost forecast
        setGraphsLSTM({}); // Clear LSTM graphs
        setGraphsXGBoost({}); // Clear XGBoost graphs
      } else {
        setCurrentPrice(`Current Stock Price: ${data.current_price}`);
        setForecastLSTM(data.lstm.forecast); // Set LSTM forecast
        setForecastXGBoost(data.xgboost.forecast); // Set XGBoost forecast

        // Append a timestamp to graph URLs to prevent caching
        const timestamp = new Date().getTime();
        setGraphsLSTM({
          actual_vs_predicted: `${data.lstm.graphs.actual_vs_predicted_lstm}?t=${timestamp}`,
          forecasted_prices: `${data.lstm.graphs.forecasted_prices_lstm}?t=${timestamp}`,
          training_vs_validation_loss: `${data.lstm.graphs.training_vs_validation_loss_lstm}?t=${timestamp}`,
          residuals_histogram: `${data.lstm.graphs.residuals_histogram_lstm}?t=${timestamp}`,
        });
        setGraphsXGBoost({
          actual_vs_predicted: `${data.xgboost.graphs.actual_vs_predicted_xgboost}?t=${timestamp}`,
          forecasted_prices: `${data.xgboost.graphs.forecasted_prices_xgboost}?t=${timestamp}`,
          training_vs_validation_loss: `${data.xgboost.graphs.training_vs_validation_loss_xgboost}?t=${timestamp}`,
          residuals_histogram: `${data.xgboost.graphs.residuals_histogram_xgboost}?t=${timestamp}`,
        });
      }
    } catch (error) {
      setError(`Error fetching data: ${error.message}`); // Set detailed error message
      setCurrentPrice(''); // Clear current price
      setForecastLSTM([]); // Clear LSTM forecast
      setForecastXGBoost([]); // Clear XGBoost forecast
      setGraphsLSTM({}); // Clear LSTM graphs
      setGraphsXGBoost({}); // Clear XGBoost graphs
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-blue-100 to-blue-300 flex flex-col items-center py-10">
      <h2 className="text-4xl font-bold text-gray-800 mb-6">Stock Price Prediction</h2>
      <p className="text-lg text-gray-700 mb-8">Using Machine Learning</p>
      <div className="flex flex-col items-center w-full max-w-md bg-white p-6 rounded-lg shadow-lg">
        <label className="text-gray-700 font-semibold mb-2 w-full text-left">
          Stock Symbol:
        </label>
        <input
          type="text"
          placeholder="Enter stock symbol (e.g., AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
        />
        <label className="text-gray-700 font-semibold mb-2 w-full text-left">
          Number of Days to Forecast:
        </label>
        <input
          type="text"
          placeholder="Enter number of days (e.g., 7)"
          value={forecastDays}
          onChange={(e) => setForecastDays(Number(e.target.value))}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
        />
        <button
          onClick={handlePredict}
          className={`w-full px-4 py-2 text-white font-semibold rounded-lg shadow-md ${
            loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 cursor-pointer hover:bg-blue-600'
          }`}
          disabled={loading}
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {/* Display error message */}
      {error && (
        <div className="mt-6 text-center text-red-600 font-semibold">
          {error}
        </div>
      )}

      {/* Display current price */}
      {!error && currentPrice && (
        <div className="mt-6 text-center text-gray-700 text-xl font-semibold">
          {currentPrice}
        </div>
      )}

      {/* Display forecasted prices in a table */}
      {forecastLSTM.length > 0 && forecastXGBoost.length > 0 && !error && (
        <div className="mt-6 w-full max-w-4xl">
          <h3 className="text-lg font-semibold mb-4 text-center">Forecasted Stock Prices for next {forecastDays} days</h3>
          <table className="w-full bg-white rounded-lg shadow-md overflow-hidden">
            <thead className="bg-blue-500 text-white">
              <tr>
                <th className="px-4 py-2 text-left border-r border-gray-600">Date</th>
                <th className="px-4 py-2 text-left border-r border-gray-600">Forecast by LSTM</th>
                <th className="px-4 py-2 text-left">Forecast by XGBoost</th>
              </tr>
            </thead>
            <tbody>
              {forecastLSTM.map((item, index) => (
                <tr key={index} className="border-b">
                  <td className="px-4 py-2 border-r border-gray-600">{item.date.split(' ')[0]}</td>
                  <td className="px-4 py-2 border-r border-gray-600">{item.value}</td>
                  <td className="px-4 py-2">{forecastXGBoost[index]?.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Display LSTM graphs */}
      {!error && (
        <div className="mt-10 grid grid-cols-1 gap-8 w-full max-w-4xl">
          {graphsLSTM.actual_vs_predicted && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">LSTM: Actual vs Predicted Stock Prices</h3>
              <img
                src={`http://localhost:5000/graph/${graphsLSTM.actual_vs_predicted}`}
                alt="LSTM Actual vs Predicted"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {graphsLSTM.forecasted_prices && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">LSTM: Forecasted Stock Prices</h3>
              <img
                src={`http://localhost:5000/graph/${graphsLSTM.forecasted_prices}`}
                alt="LSTM Forecasted Prices"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {graphsLSTM.training_vs_validation_loss && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">LSTM: Training vs Validation Loss</h3>
              <img
                src={`http://localhost:5000/graph/${graphsLSTM.training_vs_validation_loss}`}
                alt="LSTM Training vs Validation Loss"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {graphsLSTM.residuals_histogram && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">LSTM: Residuals Histogram</h3>
              <img
                src={`http://localhost:5000/graph/${graphsLSTM.residuals_histogram}`}
                alt="LSTM Residuals Histogram"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </div>
      )}

      {/* Display XGBoost graphs */}
      {!error && (
        <div className="mt-10 grid grid-cols-1 gap-8 w-full max-w-4xl">
          {graphsXGBoost.actual_vs_predicted && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">XGBoost: Actual vs Predicted Stock Prices</h3>
              <img
                src={`http://localhost:5000/graph/${graphsXGBoost.actual_vs_predicted}`}
                alt="XGBoost Actual vs Predicted"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {graphsXGBoost.forecasted_prices && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">XGBoost: Forecasted Stock Prices</h3>
              <img
                src={`http://localhost:5000/graph/${graphsXGBoost.forecasted_prices}`}
                alt="XGBoost Forecasted Prices"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {graphsXGBoost.residuals_histogram && (
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">XGBoost: Residuals Histogram</h3>
              <img
                src={`http://localhost:5000/graph/${graphsXGBoost.residuals_histogram}`}
                alt="XGBoost Residuals Histogram"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;