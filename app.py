import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Check for locally stored data, else fetch it
if os.path.exists("sp500.csv"):
    print("Loading data from local CSV file...")
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    print("Downloading data from Yahoo Finance...")
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")
    print("Data downloaded and saved as 'sp500.csv'.")

# Prepare data
print("Preparing data...")
sp500.index = pd.to_datetime(sp500.index, utc=True)
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create 'Tomorrow' and 'Target' columns for prediction
print("Creating target columns for prediction...")
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Restrict data to more recent times for simplicity
print("Slicing data from 1990-01-01...")
sp500 = sp500.loc["1990-01-01":].copy()

# Add lagged features to incorporate past price movements
print("Adding lagged features...")
for lag in range(1, 6):  # 1 to 5 days lag
    sp500[f"Close_Lag_{lag}"] = sp500["Close"].shift(lag)

# Drop rows with NaN values resulting from lagged features
sp500.dropna(inplace=True)

# Model setup
print("Setting up the Random Forest model...")
model = RandomForestClassifier(n_estimators=150, min_samples_split=10, random_state=1, max_depth=10)

# Split into training and test datasets
print("Splitting data into training and test sets...")
train = sp500[:-100]
test = sp500[-100:]

# Define predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]
# Include the lagged features
predictors += [f"Close_Lag_{lag}" for lag in range(1, 6)]

# Scale the features for better model performance
scaler = StandardScaler()
print("Scaling features...")
train.loc[:, predictors] = scaler.fit_transform(train[predictors]).astype('float64')  # Use .loc to avoid warnings
test.loc[:, predictors] = scaler.transform(test[predictors]).astype('float64')        # Use .loc to avoid warnings

# Train model
print("Training the model...")
model.fit(train[predictors], train["Target"])
print("Model training complete.")

# Make predictions
print("Making predictions on the test set...")
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Combine predictions with actual results
combined = pd.concat([test["Target"], preds], axis=1)

# Prediction function
def predict(train, test, predictors, model):
    print("Fitting model on the training set...")
    model.fit(train[predictors], train["Target"])
    print("Predicting on the test set...")
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    print("Starting backtest...")
    for i in range(start, data.shape[0], step):
        print(f"Backtesting from row {i}...")
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    print("Backtesting complete.")
    return pd.concat(all_predictions)

# Perform backtest
predictions = backtest(sp500, model, predictors)

# Calculate the accuracy of the model
accuracy = accuracy_score(predictions["Target"], predictions["Predictions"])
print(f"Model Accuracy: {accuracy * 100:.2f}%")
