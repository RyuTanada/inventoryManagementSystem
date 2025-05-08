# ml_server/train_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

# Load training data
train_data = pd.read_csv('Walmart_Train_Cleaned.csv', sep=';')

# Features and target
features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
target = 'Weekly_Sales'

X = train_data[features]
y = train_data[target]

# --- Feature Engineering: Add Polynomial Features ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Save Polynomial Transformer
joblib.dump(poly, 'poly_transformer.pkl')

# --- Scale the features ---
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
joblib.dump(scaler, 'scaler.pkl')

# --- Train Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_poly_scaled, y)
joblib.dump(lr_model, 'linear_regression.pkl')

# --- Train Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_poly, y)
joblib.dump(rf_model, 'random_forest.pkl')

# --- Train LSTM ---
X_lstm = np.reshape(X_poly_scaled, (X_poly_scaled.shape[0], 1, X_poly_scaled.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(units=64, input_shape=(1, X_poly_scaled.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss=MeanSquaredError())

lstm_model.fit(X_lstm, y, epochs=20, batch_size=32, verbose=1)
lstm_model.save('lstm_model.h5')

print("Models trained and saved successfully!")