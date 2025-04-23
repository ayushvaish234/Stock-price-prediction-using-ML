import React, { useState } from 'react';

function App() {
  const [symbol, setSymbol] = useState('');
  const [stockInfo, setStockInfo] = useState(null);
  const [forecastDays, setForecastDays] = useState(7); // State for forecast days
  const [currentPrice, setCurrentPrice] = useState('');
  const [forecastLSTM, setForecastLSTM] = useState([]); // Forecast by LSTM
  const [forecastXGBoost, setForecastXGBoost] = useState([]); // Forecast by XGBoost
  const [forecastCombined, setForecastCombined] = useState([]);
  const [graphsLSTM, setGraphsLSTM] = useState({}); // Graphs for LSTM
  const [graphsXGBoost, setGraphsXGBoost] = useState({}); // Graphs for XGBoost
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(''); // State to store error messages

  document.title = 'Stock Price Prediction';

  const fetchStockInfo = async () => {
    try {
      const response = await fetch('http://localhost:5000/stock-info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol }),
      });
      const data = await response.json();
  
      if (data.error) {
        setError(data.error);
        setStockInfo(null);
      } else {
        setStockInfo(data);
        setError(''); // Clear any previous error
      }
    } catch (error) {
      setError(`Error fetching stock info: ${error.message}`);
      setStockInfo(null);
    }
  };
  function AboutCompanyText({ text }) {
    const [isExpanded, setIsExpanded] = useState(false);
  
    return (
      <div>
        <p className="text-sm text-gray-700 text-justify">
          {isExpanded ? text : `${text.slice(0, 300)}...`}
        </p>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-blue-500 font-semibold mt-2"
        >
          {isExpanded ? 'Read Less' : 'Read More'}
        </button>
      </div>
    );
  }
  const handlePredict = async () => {
    setLoading(true);
    setStockInfo(null); 
    await fetchStockInfo(); 
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
        setForecastCombined(data.combined_forecast.forecast);
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
<div className="min-h-screen bg-blue-500 flex flex-col items-center py-0">
  <header className="w-full bg-white text-center p-4 mb-6  shadow-md">
    <h2 className="text-4xl font-bold text-blue-500">Stock Price Prediction</h2>
    <p className="text-sm text-blue-500">Using Machine Learning</p>
  </header>

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
  <div className="mt-9 text-center bg-green-500 text-white text-2xl font-extrabold py-4 px-6 rounded-lg shadow-lg">
    {currentPrice}
  </div>
)}

      {/* Display forecasted prices in a table */}
      {forecastLSTM.length > 0 && forecastXGBoost.length > 0 && forecastCombined.length > 0  && !error && (
        <div className="mt-6 w-full max-w-4xl">
          <h3 className="text-xl font-bold text-white mb-4 text-left">FORECASTED STOCK PRICE FOR NEXT {forecastDays} DAYS</h3>

          <table className="w-full bg-white rounded-lg shadow-md overflow-hidden">
            <thead className="bg-white text-blue-500 border-b border-gray-600">
              <tr>
                <th className="px-4 py-2 text-left border-r border-gray-600">Date</th>
                <th className="px-4 py-2 text-left border-r border-gray-600">Forecast by LSTM</th>
                <th className="px-4 py-2 text-left border-r border-gray-600">Forecast by XGBoost</th>
                <th className="px-4 py-2 text-left">Final Forecast</th>
              </tr>
            </thead>
            <tbody>
              {forecastLSTM.map((item, index) => (
                <tr key={index} className="border-b">
                  <td className="px-4 py-2 border-r border-gray-600">{item.date.split(' ')[0]}</td>
                  <td className="px-4 py-2 border-r border-gray-600">{item.value}</td>
                  <td className="px-4 py-2 border-r border-gray-600">{forecastXGBoost[index]?.value}</td>
                  <td className="px-4 py-2">{forecastCombined[index]?.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
{stockInfo && !error && (
  <div className="mt-6 w-full flex justify-center px-4">
    <div className="w-full max-w-6xl flex flex-wrap md:flex-nowrap justify-between gap-6">
      {/* Left Card: Stock Information */}
      <div className="w-full md:w-[48%] bg-white p-4 rounded-lg shadow-lg">
        <h3 className="text-xl font-bold text-blue-500 mb-2">Stock Information</h3>
        <p className="text-sm"><strong>Company Name:</strong> {stockInfo.name}</p>
        <p className="text-sm"><strong>Symbol:</strong> {stockInfo.symbol}</p>
        <p className="text-sm"><strong>Exchange:</strong> {stockInfo.exchange}</p>
        <p className="text-sm"><strong>Sector:</strong> {stockInfo.sector}</p>
        <p className="text-sm"><strong>Industry:</strong> {stockInfo.industry}</p>
        <p className="text-sm"><strong>Market Cap:</strong> {stockInfo.marketCap}</p>
        <p className="text-sm"><strong>P/E Ratio:</strong> {stockInfo.peRatio}</p>
        <p className="text-sm"><strong>All-Time High:</strong> {stockInfo.allTimeHigh}</p>
        <p className="text-sm"><strong>All-Time Low:</strong> {stockInfo.allTimeLow}</p>
      </div>

      {/* Right Card: About Company */}
      <div className="w-full md:w-[48%] bg-white p-4 rounded-lg shadow-lg">
        <h3 className="text-xl font-bold text-blue-500 mb-2">About Company</h3>
        <AboutCompanyText text={stockInfo.about} />
        </div>
    </div>
  </div>
)}
 {/* LSTM Section */}
 
 {forecastLSTM.length > 0 && forecastXGBoost.length > 0 && forecastCombined.length > 0  && !error && (
      
      <div className="mt-10 w-full  border-t border-b border-gray-100 rounded-lg" style={{ backgroundColor: '#f1f1f8' }}>
        
      <div className="w-full max-w-4xl mx-auto">
        <h3 className="text-xl font-bold text-Black-500 mb-4 text-left uppercase mt-6">
           LSTM: Actual vs Predicted Stock Prices
    </h3>
    <div className="bg-white p-4 rounded-lg shadow-md mb-6">
      <img
        src={`http://localhost:5000/graph/${graphsLSTM.actual_vs_predicted}`}
        alt="LSTM Actual vs Predicted"
        className="w-full rounded-lg"
      />
    </div>

    {graphsLSTM.forecasted_prices && (
      <>
        <h3 className="text-xl font-bold text-Black mb-4 text-left uppercase">
          LSTM: Forecasted Stock Prices
        </h3>
        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <img
            src={`http://localhost:5000/graph/${graphsLSTM.forecasted_prices}`}
            alt="LSTM Forecasted Prices"
            className="w-full rounded-lg"
          />
        </div>
      </>
    )}

    {graphsLSTM.training_vs_validation_loss && (
      <>
        <h3 className="text-xl font-bold text-Black mb-4 text-left uppercase">
          LSTM: Training vs Validation Loss
        </h3>
        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <img
            src={`http://localhost:5000/graph/${graphsLSTM.training_vs_validation_loss}`}
            alt="LSTM Training vs Validation Loss"
            className="w-full rounded-lg"
          />
        </div>
      </>
    )}

    {graphsLSTM.residuals_histogram && (
      <>
        <h3 className="text-xl font-bold text-Black mb-4 text-left uppercase">
          LSTM: Residuals Histogram
        </h3>
        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <img
            src={`http://localhost:5000/graph/${graphsLSTM.residuals_histogram}`}
            alt="LSTM Residuals Histogram"
            className="w-full rounded-lg"
          />
        </div>
      </>
    )}
        </div>
  </div>
)}
  {/* Footer */}
  <footer className="w-full bg-gray-800 text-white text-center py-4 mt-10">
    <p >
      Stock Price Prediction
    </p>
    <p className="text-sm">project by</p>
    <p className="text-sm">Ayush vaish | Sudhanshu Tiwari | Anil Kacchap | Aman kumar maourya</p>
    <p className="text-sm"> 230231017 | 230231060 | 230231011 | 230231008     </p>
  </footer>
</div>
  
  );
}

export default App;