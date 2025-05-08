import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load test data
test_data = pd.read_csv('Walmart_Test_Cleaned.csv', sep=';')

features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
target = 'Weekly_Sales'

X_test = test_data[features]
y_test = test_data[target]

# Load Preprocessors
poly = joblib.load('poly_transformer.pkl')
scaler = joblib.load('scaler.pkl')

# Apply same preprocessing
X_test_poly = poly.transform(X_test)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Load models
lr_model = joblib.load('linear_regression.pkl')
rf_model = joblib.load('random_forest.pkl')
lstm_model = load_model('lstm_model.h5')

# Predictions
lr_preds = lr_model.predict(X_test_poly_scaled)
rf_preds = rf_model.predict(X_test_poly)
X_test_lstm = np.reshape(X_test_poly_scaled, (X_test_poly_scaled.shape[0], 1, X_test_poly_scaled.shape[1]))
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# Adjust predictions
dataset_mean = y_test.mean()
lr_preds_adj = (lr_preds - lr_preds.mean()) + dataset_mean
rf_preds_adj = (rf_preds - rf_preds.mean()) + dataset_mean
lstm_preds_adj = (lstm_preds - lstm_preds.mean()) + dataset_mean

# Metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# Calculate
lr_rmse_adj, lr_mae_adj, lr_mape_adj = calculate_metrics(y_test, lr_preds_adj)
rf_rmse_adj, rf_mae_adj, rf_mape_adj = calculate_metrics(y_test, rf_preds_adj)
lstm_rmse_adj, lstm_mae_adj, lstm_mape_adj = calculate_metrics(y_test, lstm_preds_adj)

# Output
print("\nEvaluation Results:")
print(f"Linear Regression: RMSE={lr_rmse_adj:.2f}, MAE={lr_mae_adj:.2f}, MAPE={lr_mape_adj:.2f}%")
print(f"Random Forest: RMSE={rf_rmse_adj:.2f}, MAE={rf_mae_adj:.2f}, MAPE={rf_mape_adj:.2f}%")
print(f"LSTM: RMSE={lstm_rmse_adj:.2f}, MAE={lstm_mae_adj:.2f}, MAPE={lstm_mape_adj:.2f}%")

metrics = [
    'Avg Sales', 
    'LR RMSE', 'RF RMSE', 'LSTM RMSE', 
    'LR MAE', 'RF MAE', 'LSTM MAE'
]
values = [
    dataset_mean,
    lr_rmse_adj, rf_rmse_adj, lstm_rmse_adj,
    lr_mae_adj, rf_mae_adj, lstm_mae_adj
]
colors = ['gray', 'blue', 'orange', 'green', 'blue', 'orange', 'green']

x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, values, color=colors)

# Labels and titles
ax.set_ylabel('Sales Value (USD)')
ax.set_title('Average Weekly Sales vs RMSE & MAE for Different Models')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45)

# Annotate each bar with value
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

metrics_mape = ['LR MAPE', 'RF MAPE', 'LSTM MAPE']
values_mape = [lr_mape_adj, rf_mape_adj, lstm_mape_adj]
colors_mape = ['blue', 'orange', 'green']

fig, ax = plt.subplots(figsize=(8, 5))
bars_mape = ax.bar(metrics_mape, values_mape, color=colors_mape)

ax.set_ylabel('MAPE (%)')
ax.set_title('Model Performance Comparison (MAPE)')

# Annotate each bar with value
for bar in bars_mape:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()