//Author :Aman Kumar Maourya
//Roll No: 230231008

import React, { useState } from 'react';
import './App.css';
function App() {
  const [symbol, setSymbol] = useState('');
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  document.title = 'Stock Price Prediction using machine learning'; 
  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol }),
      });
      const data = await response.json();
      setPrediction(` Current Price: ${data.current_price} | Forecast: ${data.forecast.join(', ')} `);
    } catch (error) {
      setPrediction('Error fetching prediction');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Stock Price Prediction</h2>
      <p>using ML</p>
      <input
        type="text"
        placeholder="Enter stock symbol (e.g., AAPL)"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        style={{ padding: '0.5rem', marginRight: '1rem' }}
      />
      <button onClick={handlePredict} style={{ padding: '0.5rem 1rem' }}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      <div style={{ marginTop: '1rem' }}>{prediction}</div>
    </div>
  );
}

export default App;
