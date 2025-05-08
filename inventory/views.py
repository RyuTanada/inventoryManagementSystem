import pandas as pd
import numpy as np
import joblib
import locale
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from io import StringIO

# ✅ Set locale for comma formatting (e.g., 1,000,000.00)
try:
    locale.setlocale(locale.LC_ALL, '')  # Automatically use system locale
except locale.Error:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # Fallback if system fails

# Global to store predictions for download
latest_predictions_df = None

@csrf_exempt
def home(request):
    global latest_predictions_df
    predictions = None
    error = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']

            try:
                df = pd.read_csv(csv_file, sep=';')

                expected_cols = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
                if not all(col in df.columns for col in expected_cols):
                    raise ValueError("CSV must contain columns: IsHoliday, Temperature, Fuel_Price, Unemployment")

                # Ensure dates are in correct order
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.sort_values(by='Date')

                # Load model artifacts
                poly = joblib.load('ml_server/poly_transformer.pkl')
                scaler = joblib.load('ml_server/scaler.pkl')
                model = joblib.load('ml_server/linear_regression.pkl')

                # Prepare inputs
                X = df[expected_cols]
                X_poly = poly.transform(X)
                X_scaled = scaler.transform(X_poly)
                preds = model.predict(X_scaled)

                # Round predictions and format with comma
                df['Predicted_Weekly_Sales'] = preds.round(2)
                df['Predicted_Weekly_Sales_Display'] = df['Predicted_Weekly_Sales'].apply(
                    lambda x: locale.format_string('%.2f', x, grouping=True)
                )

                # Extract required columns for frontend table
                latest_predictions_df = df[['Store', 'Date', 'Predicted_Weekly_Sales_Display']]
                predictions = latest_predictions_df.values.tolist()

            except Exception as e:
                error = str(e)

        elif 'download' in request.POST:
            if latest_predictions_df is not None:
                buffer = StringIO()
                latest_predictions_df.to_csv(buffer, index=False)
                buffer.seek(0)
                response = HttpResponse(buffer, content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename=predictions.csv'
                return response
            else:
                error = "No predictions available to download."

    return render(request, 'home.html', {'predictions': predictions, 'error': error})