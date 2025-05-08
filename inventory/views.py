import pandas as pd
import joblib
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model

@csrf_exempt
def home(request):
    print("✅ Request received")
    predictions = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        print("✅ CSV file detected")
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file, sep=';')
            print("✅ CSV content:\n", df.head())
            # Ensure correct columns
            expected_cols = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
            if not all(col in df.columns for col in expected_cols):
                raise ValueError("CSV must contain columns: IsHoliday, Temperature, Fuel_Price, Unemployment")

            # Load preprocessing tools and model
            poly = joblib.load('/Users/ryutanada/Developer/inventoryManagementSystem/ml_server/poly_transformer.pkl')
            scaler = joblib.load('/Users/ryutanada/Developer/inventoryManagementSystem/ml_server/scaler.pkl')
            model = joblib.load('/Users/ryutanada/Developer/inventoryManagementSystem/ml_server/linear_regression.pkl')

            # Apply same transformations
            X = df[expected_cols]
            X_poly = poly.transform(X)
            X_scaled = scaler.transform(X_poly)
            preds = model.predict(X_scaled)

            # Format predictions
            df['Predicted Weekly Sales'] = preds
            predictions = list(zip(X.values.tolist(), preds.round(2)))
            print("✅ Predictions complete")

        except Exception as e:
            print("❌ Error:", e)
            return render(request, 'home.html', {'error': str(e)})

    return render(request, 'home.html', {'predictions': predictions})