# Stock Market Prediction Model

This project implements a stock market prediction model using machine learning techniques, specifically a Random Forest Classifier. The model predicts whether the closing price of the S&P 500 index will increase the next day based on historical data and various technical indicators.

## Table of Contents
- [Model Overview](#model-overview)
- [Features Used](#features-used)
- [How to Run the Model](#how-to-run-the-model)
- [Requirements](#requirements)
- [License](#license)

## Model Overview

The model follows these key steps:
1. **Data Collection**: It downloads historical S&P 500 data using the `yfinance` library, or loads it from a local CSV file if available.
2. **Data Preprocessing**: The data is cleaned by removing irrelevant columns and creating a target variable that indicates whether the stock price will increase the next day.
3. **Feature Engineering**: Various features are calculated, including:
   - Rolling averages
   - Momentum indicators
   - Volatility
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
   - Lagged values for past prices and volumes
4. **Model Training**: A Random Forest Classifier is trained using the generated features to predict the target variable.
5. **Backtesting**: The model is tested on historical data, and its predictions are evaluated for accuracy.

## Features Used

The following features are utilized in the model:
- Closing price, volume, open, high, and low prices.
- Technical indicators: RSI, MACD.
- Rolling metrics: Close ratio, volatility, and momentum over various horizons.
- Lagged features for the closing price, volume, high, and low values.

## How to Run the Model

To run the model, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. **Install the required packages: Make sure you have Python 3.x installed. Install the necessary packages using pip:**
    pip install yfinance pandas numpy scikit-learn
3. **Run the script: Execute the Python script to run the model:**
    python stock_market_prediction.py

## Requirements
* Python 3.x
* yfinance library for fetching stock data.
* pandas for data manipulation.
* numpy for numerical computations.
* scikit-learn for machine learning functionalities.

