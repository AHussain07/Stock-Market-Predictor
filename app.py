import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load S&P 500 data from a CSV if available, or download it using yfinance
if os.path.exists("sp500.csv"):
    print("Loading S&P 500 data from CSV...")
    sp500_data = pd.read_csv("sp500.csv", index_col=0)
else:
    print("Downloading S&P 500 data using yfinance...")
    sp500_ticker = yf.Ticker("^GSPC")
    sp500_data = sp500_ticker.history(period="max")
    sp500_data.to_csv("sp500.csv")

# Preprocess data
sp500_data.index = pd.to_datetime(sp500_data.index, utc=True)
sp500_data.drop(columns=["Dividends", "Stock Splits"], inplace=True)

# Create target variable (1 if tomorrow's closing price is higher than today's, else 0)
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)

# Use more recent data to reduce dataset size and speed up training
sp500_data = sp500_data.loc["2000-01-01":].copy()

# Feature engineering: Add rolling averages, momentum indicators, volatility, and technical indicators
rolling_horizons = [2, 5, 10, 21, 60]
for horizon in rolling_horizons:
    rolling_avg = sp500_data["Close"].rolling(window=horizon).mean()
    sp500_data[f"Close_Ratio_{horizon}"] = sp500_data["Close"] / rolling_avg
    sp500_data[f"Volatility_{horizon}"] = sp500_data["Close"].rolling(window=horizon).std()
    sp500_data[f"Momentum_{horizon}"] = sp500_data["Close"].diff(horizon)

# Calculate the Relative Strength Index (RSI)
def compute_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

sp500_data["RSI_14"] = compute_rsi(sp500_data["Close"])

# Calculate MACD (12-day EMA - 26-day EMA)
ema_12 = sp500_data["Close"].ewm(span=12, adjust=False).mean()
ema_26 = sp500_data["Close"].ewm(span=26, adjust=False).mean()
sp500_data["MACD"] = ema_12 - ema_26

# Lag features: Adding past close, volume, high, and low values
num_lags = 3  # Number of lags to include
for lag in range(1, num_lags + 1):
    sp500_data[f"Close_Lag_{lag}"] = sp500_data["Close"].shift(lag)
    sp500_data[f"Volume_Lag_{lag}"] = sp500_data["Volume"].shift(lag)
    sp500_data[f"High_Lag_{lag}"] = sp500_data["High"].shift(lag)
    sp500_data[f"Low_Lag_{lag}"] = sp500_data["Low"].shift(lag)

# Drop rows with NaN values that result from rolling calculations and lags
sp500_data = sp500_data.dropna()

# Define the model (Random Forest Classifier)
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=100, random_state=1, n_jobs=-1)

# Define predictors (including technical indicators, rolling metrics, and lag features)
predictors = ["Close", "Volume", "Open", "High", "Low", "RSI_14", "MACD"]
for horizon in rolling_horizons:
    predictors += [f"Close_Ratio_{horizon}", f"Volatility_{horizon}", f"Momentum_{horizon}"]

for lag in range(1, num_lags + 1):
    predictors += [f"Close_Lag_{lag}", f"Volume_Lag_{lag}", f"High_Lag_{lag}", f"Low_Lag_{lag}"]

# Train the model and compute feature importance
rf_model.fit(sp500_data[predictors], sp500_data["Target"])
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Keep top 15 important features
top_features = [predictors[i] for i in indices[:15]]

# Define prediction function
def predict(train_data, test_data, predictors, model):
    model.fit(train_data[predictors], train_data["Target"])
    predictions = model.predict(test_data[predictors])
    return pd.Series(predictions, index=test_data.index, name="Predictions")

# Define backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train_data = data.iloc[0:i].copy()
        test_data = data.iloc[i:(i + step)].copy()
        predictions = predict(train_data, test_data, predictors, model)
        combined = pd.concat([test_data["Target"], predictions], axis=1)
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Perform backtest using the top features
predictions = backtest(sp500_data, rf_model, top_features)

# Calculate accuracy score
accuracy = accuracy_score(predictions["Target"], predictions["Predictions"])

# Print the total number of trading days
total_trading_days = predictions.shape[0]
print(f"Total number of trading days: {total_trading_days}")

# Print the number of days the model said to buy
buy_days = predictions[predictions["Predictions"] == 1].shape[0]
print(f"Number of days the model said to buy: {buy_days}")

# Return the model accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%")
