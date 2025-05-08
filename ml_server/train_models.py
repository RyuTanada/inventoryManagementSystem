# ml_server/train_models.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

# Define model directory as absolute path
model_dir = os.path.dirname(__file__)

# Ensure the 'ml_server' directory exists
os.makedirs(model_dir, exist_ok=True)

# Load training data
train_data = pd.read_csv('Walmart_Train_Cleaned.csv', sep=';')

# Features and target
features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
target = 'Weekly_Sales'

X = train_data[features]
y = train_data[target]
y = np.array(y).reshape(-1, 1)  # Reshape for scaler compatibility

# --- Feature Engineering: Add Polynomial Features ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
joblib.dump(poly, os.path.join(model_dir, 'poly_transformer.pkl'))

# --- Scale the features ---
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# --- Train Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_poly_scaled, y)
joblib.dump(lr_model, os.path.join(model_dir, 'linear_regression.pkl'))

# --- Train Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_poly, y)
joblib.dump(rf_model, os.path.join(model_dir, 'random_forest.pkl'))

# --- Scale y for LSTM ---
lstm_y_scaler = MinMaxScaler()
y_scaled = lstm_y_scaler.fit_transform(y)
joblib.dump(lstm_y_scaler, os.path.join(model_dir, 'lstm_y_scaler.pkl'))
print("✅ lstm_y_scaler.pkl saved successfully!")

# --- Prepare input for LSTM ---
X_lstm = np.reshape(X_poly_scaled, (X_poly_scaled.shape[0], 1, X_poly_scaled.shape[1]))

# --- Build and Train LSTM ---
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, input_shape=(1, X_poly_scaled.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss=MeanSquaredError())

lstm_model.fit(X_lstm, y_scaled, epochs=20, batch_size=32, verbose=1)
lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))

print("✅ Models trained and saved successfully!")