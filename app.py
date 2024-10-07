import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import os

# Load S&P 500 data from a CSV if available, or download it using yfinance
if os.path.exists("sp500.csv"):
    print("Loading S&P 500 data from CSV...")
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    print("Downloading S&P 500 data using yfinance...")
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

# Preprocess data
sp500.index = pd.to_datetime(sp500.index, utc=True)
del sp500["Dividends"]
del sp500["Stock Splits"]

# Add target and slice the data
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Use more recent data to reduce dataset size and speed up training
sp500 = sp500.loc["2000-01-01":].copy()

# Feature engineering: Add rolling averages, momentum indicators, volatility, and technical indicators
horizons = [2, 5, 10, 21, 60]
for horizon in horizons:
    rolling_avg = sp500["Close"].rolling(window=horizon).mean()
    sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_avg
    sp500[f"Volatility_{horizon}"] = sp500["Close"].rolling(window=horizon).std()
    sp500[f"Momentum_{horizon}"] = sp500["Close"].diff(horizon)

# Technical Indicators: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD)
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

sp500["RSI_14"] = compute_rsi(sp500["Close"])

# MACD (12-day EMA - 26-day EMA)
ema_12 = sp500["Close"].ewm(span=12, adjust=False).mean()
ema_26 = sp500["Close"].ewm(span=26, adjust=False).mean()
sp500["MACD"] = ema_12 - ema_26

# Lag features: Adding past close, volume, high, and low values as new features
lags = 3  # Number of lags to include
for lag in range(1, lags + 1):
    sp500[f"Close_Lag_{lag}"] = sp500["Close"].shift(lag)
    sp500[f"Volume_Lag_{lag}"] = sp500["Volume"].shift(lag)
    sp500[f"High_Lag_{lag}"] = sp500["High"].shift(lag)
    sp500[f"Low_Lag_{lag}"] = sp500["Low"].shift(lag)

# Drop rows with NaN values that result from rolling calculations and lags
sp500 = sp500.dropna()

# Define the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=100, random_state=1, n_jobs=-1)

# Define predictors (now including technical indicators, rolling metrics, and lag features)
predictors = ["Close", "Volume", "Open", "High", "Low", "RSI_14", "MACD"]
for horizon in horizons:
    predictors += [f"Close_Ratio_{horizon}", f"Volatility_{horizon}", f"Momentum_{horizon}"]

for lag in range(1, lags + 1):
    predictors += [f"Close_Lag_{lag}", f"Volume_Lag_{lag}", f"High_Lag_{lag}", f"Low_Lag_{lag}"]

# Feature Selection using importance from Random Forest
model.fit(sp500[predictors], sp500["Target"])
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Keep top 15 important features
top_features = [predictors[i] for i in indices[:15]]

# Define backtesting and prediction functions
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    return pd.Series(preds, index=test.index, name="Predictions")

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        combined = pd.concat([test["Target"], predictions], axis=1)
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Perform backtest using the top features
predictions = backtest(sp500, model, top_features)

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
