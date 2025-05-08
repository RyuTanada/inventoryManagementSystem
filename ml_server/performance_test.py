import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load test dataset
test_data = pd.read_csv('Walmart_Test_Cleaned.csv', sep=';')

# Load models
lr_model = joblib.load('linear_regression.pkl')
rf_model = joblib.load('random_forest.pkl')
lstm_model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Prepare features
features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
X_test = test_data[features]
y_test = test_data['Weekly_Sales']

# Scale features for LSTM
X_test_scaled = scaler.transform(X_test)

# Predict with models
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
lstm_preds = lstm_model.predict(X_test_scaled).flatten()

# Calculate metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))

lr_mae = mean_absolute_error(y_test, lr_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
lstm_mae = mean_absolute_error(y_test, lstm_preds)

# Calculate average weekly sales
average_sales = y_test.mean()

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'LSTM'],
    'Average Weekly_Sales': [average_sales, average_sales, average_sales],
    'RMSE': [lr_rmse, rf_rmse, lstm_rmse],
    'MAE': [lr_mae, rf_mae, lstm_mae]
})

print(results_df)
results_df.to_csv('model_performance_results.csv', index=False)