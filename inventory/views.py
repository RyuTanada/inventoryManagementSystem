import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from io import StringIO
from tensorflow.keras.models import load_model
from .models import Product
from .forms import ProductForm

# Store predictions globally for CSV export
latest_predictions_df = None

@csrf_exempt
def forecast(request):
    global latest_predictions_df
    predictions = None
    error = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']

            try:
                df = pd.read_csv(csv_file, sep=';')

                print("🔍 CSV columns:", df.columns.tolist())

                expected_cols = ['IsHoliday', 'Temperature', 'Fuel_Price', 'Unemployment']
                if not all(col in df.columns for col in expected_cols):
                    raise ValueError("CSV must contain columns: IsHoliday, Temperature, Fuel_Price, Unemployment")

                # Sort chronologically by Date (if exists)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.sort_values(by='Date')

                # Load preprocessing transformers
                poly = joblib.load('ml_server/poly_transformer.pkl')
                scaler = joblib.load('ml_server/scaler.pkl')

                # Feature preparation
                X = df[expected_cols]
                X_poly = poly.transform(X)
                X_scaled = scaler.transform(X_poly)

                # Load models
                lr_model = joblib.load('ml_server/linear_regression.pkl')  # Short-term
                rf_model = joblib.load('ml_server/random_forest.pkl')      # Seasonal
                lstm_model = load_model('ml_server/lstm_model.h5', compile=False)         # Long-term model
                lstm_y_scaler = joblib.load('ml_server/lstm_y_scaler.pkl') # Long-term target scaler

                # Prepare LSTM input shape [samples, timesteps, features]
                X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

                # Make predictions
                short_preds = lr_model.predict(X_scaled).round(2)
                seasonal_preds = rf_model.predict(X_poly).round(2)

                # Predict with LSTM and inverse scale
                raw_preds = lstm_model.predict(X_lstm)
                long_preds = lstm_y_scaler.inverse_transform(raw_preds).flatten().round(2)

                print("✅ Predictions generated:", predictions)
                print("🔍 LSTM raw_preds:", raw_preds[:5])
                print("🔍 Inversed LSTM:", long_preds[:5])

                # Append predictions to dataframe
                df['Predicted Short-Term Sales'] = short_preds
                df['Predicted Seasonal Sales'] = seasonal_preds
                df['Predicted Long-Term Sales'] = long_preds

                # Save prediction subset for frontend and download
                cols = ['Date', 'Predicted Short-Term Sales', 'Predicted Seasonal Sales', 'Predicted Long-Term Sales']
                if 'Store' in df.columns:
                    cols.insert(0, 'Store')
                latest_predictions_df = df[cols]
                predictions = latest_predictions_df.values.tolist()

            except Exception as e:
                error = str(e)
                print("❌ Error during processing:", error)

        elif 'download' in request.POST:
            if latest_predictions_df is not None:
                buffer = StringIO()
                latest_predictions_df.to_csv(buffer, index=False)
                buffer.seek(0)
                response = HttpResponse(buffer, content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename=predictions.csv'
                return response
            else:
                error = "No predictions to download."

    return render(request, 'forecast.html', {'predictions': predictions, 'error': error})

def dashboard(request):
    return render(request, 'home.html')

def inventory_list(request):
    products = Product.objects.all()
    return render(request, 'inventory_list.html', {'products': products})

def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('inventory_list')
    else:
        form = ProductForm()
    return render(request, 'add_product.html', {'form': form})

def edit_product(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            form.save()
            return redirect('inventory_list')
    else:
        form = ProductForm(instance=product)
    return render(request, 'edit_product.html', {'form': form, 'product': product})

def delete_product(request, pk):
    product = get_object_or_404(Product, pk=pk)
    product.delete()
    return redirect('inventory_list')