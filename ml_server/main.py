from fastapi import FastAPI
import pandas as pd
import joblib
import tensorflow as tf

app = FastAPI()

# Load your saved models
lr_model = joblib.load("linear_regression.pkl")
rf_model = joblib.load("random_forest.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def predict(features: dict):
    df = pd.DataFrame([features])

    lr_pred = lr_model.predict(df)
    rf_pred = rf_model.predict(df)
    lstm_pred = lstm_model.predict(df)

    return {
        "linear_regression": lr_pred.tolist(),
        "random_forest": rf_pred.tolist(),
        "lstm": lstm_pred.tolist()
    }