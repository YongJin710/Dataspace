import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df[['Close']].replace('[\$,]', '', regex=True).astype(float)  # Ensure numeric values
    df = df.sort_index()
    df = df.asfreq('D')  # Ensure a daily frequency
    df.fillna(method='ffill', inplace=True)  # Fill missing values if any
    return df

# Preprocessing
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Prepare sequences for LSTM/GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define LSTM model
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Define GRU model
def build_gru(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        GRU(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess data
file_path = '../Prediction of Tesla Stock Price/HistoricalData_1726367135218.csv'  # Update with actual file path
df = load_data(file_path)
scaled_data, scaler = preprocess_data(df)

# Prepare sequences
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Train LSTM
lstm_model = build_lstm((seq_length, 1))
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
lstm_preds = lstm_model.predict(X_test)

# Train GRU
gru_model = build_gru((seq_length, 1))
gru_model.fit(X_train, y_train, epochs=10, batch_size=32)
gru_preds = gru_model.predict(X_test)

# Train XGBoost
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train_flat, y_train)
xgb_preds = xgb_model.predict(X_test_flat)

# Inverse transform predictions
lstm_preds = scaler.inverse_transform(lstm_preds)
gru_preds = scaler.inverse_transform(gru_preds)
xgb_preds = scaler.inverse_transform(xgb_preds.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

# Calculate RMSE and MSE
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    return rmse, mse

lstm_rmse, lstm_mse = calculate_metrics(y_test_actual, lstm_preds)
gru_rmse, gru_mse = calculate_metrics(y_test_actual, gru_preds)
xgb_rmse, xgb_mse = calculate_metrics(y_test_actual, xgb_preds)

# Combine train and test predictions for a complete timeline
full_dates = df.index[seq_length:]  # Offset for sequence length
full_actual = np.concatenate([y_train_actual, y_test_actual])
full_lstm = np.concatenate([np.full_like(y_train_actual, np.nan), lstm_preds])  # Align to timeline
full_gru = np.concatenate([np.full_like(y_train_actual, np.nan), gru_preds])
full_xgb = np.concatenate([np.full_like(y_train_actual, np.nan), xgb_preds])

plt.figure(figsize=(12, 6))
plt.plot(full_dates, full_actual, label='Actual Price', color='black')
plt.plot(full_dates, full_lstm, label=f'LSTM (RMSE: {lstm_rmse:.2f}, MSE: {lstm_mse:.2f})', linestyle='dashed')
plt.plot(full_dates, full_gru, label=f'GRU (RMSE: {gru_rmse:.2f}, MSE: {gru_mse:.2f})', linestyle='dotted')
plt.plot(full_dates, full_xgb, label=f'XGBoost (RMSE: {xgb_rmse:.2f}, MSE: {xgb_mse:.2f})', linestyle='dashdot')

plt.legend()
plt.title('Bitcoin Price Prediction with LSTM, GRU, and XGBoost')
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()
