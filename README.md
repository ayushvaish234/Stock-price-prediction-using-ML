# Stock Price Prediction Using Machine Learning

This project is a web application that predicts future stock prices using advanced machine learning models. It provides real-time stock data, detailed company information, and forecasts using **LSTM (Long Short-Term Memory)** and **XGBoost** models. The app also generates insightful graphs to evaluate model performance and predictions.

---

## Features

### 1. **Real-Time Stock Data**
- Fetches live stock data using **Yahoo Finance API**.
- Displays key stock information:
  - Company Name
  - Symbol
  - Exchange
  - Sector
  - Industry
  - Market Cap
  - P/E Ratio
  - 52-Week High/Low
  - About the Company

### 2. **Stock Price Forecasting**
- Predicts future stock prices using:
  - **LSTM**: A deep learning model for sequential data.
  - **XGBoost**: A gradient boosting algorithm for accurate predictions.
- Combines predictions from both models using a weighted average for a **final forecast**.

### 3. **Graphs**
- Generates graphs for:
  - Actual vs Predicted Prices
  - Forecasted Stock Prices
  - Training vs Validation Loss
  - Residuals Histogram
- Graphs are dynamically updated for each prediction.

### 4. **User-Friendly Interface**
- Built with **React** and styled using **TailwindCSS** for a modern and responsive design.
- Displays error messages and loading states for better user experience.

---

## Technologies Used

### **Frontend**
- **React**: For building the user interface.
- **TailwindCSS**: For responsive and modern styling.

### **Backend**
- **Flask**: For handling API requests and serving data.
- **pandas** and **numpy**: For data manipulation and preprocessing.
- **scikit-learn**: For scaling and preparing data.
- **tensorflow**: For building and training the LSTM model.
- **xgboost**: For building the XGBoost model.
- **matplotlib**: For generating graphs.
- **yfinance**: For fetching real-time stock data.

---

## How It Works

1. **Input Stock Symbol:**
   - Enter the stock symbol (e.g., AAPL for Apple Inc.).
   - Specify the number of days to forecast.

2. **Fetch Stock Data:**
   - The app retrieves historical stock data for the past two years.

3. **Train Models:**
   - The backend trains the **LSTM** and **XGBoost** models on the fetched data.

4. **Generate Predictions:**
   - Both models predict future stock prices.
   - A weighted average of the predictions is calculated for the final forecast.

5. **Display Results:**
   - The app displays:
     - Current stock price.
     - Forecasted prices for the specified number of days.
     - Graphs for model performance and predictions.

---

## How to Run Locally

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Backend Dependencies
```bash
cd backend
pip install flask pandas numpy scikit-learn tensorflow xgboost matplotlib yfinance
```

### 3. Install Frontend Dependencies
```bash
cd ../frontend
npm install
```

### 4. Start the Application
Start backend:
```bash
cd ../backend
python app.py
```

Start frontend:
```bash
cd ../frontend
npm run dev
```

Open your browser and visit: [http://localhost:5173](http://localhost:5173)

---

## 👥 Contributors
- **Ayush Vaish**
- **Sudhanshu Tiwari**
- **Anil Kacchap**
- **Aman Kumar Maourya**

---
