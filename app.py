import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load model & scaler
clv_model = joblib.load("clv_model.pkl")    # Regression Model
clv_scaler = joblib.load("clv_scaler.pkl")  # StandardScaler for CLV

FEATURES = [
    "Quantity", "UnitPrice", "Frequency", "Recency",
    "Monetary", "CountryEncoded"
]

def predict_clv(quantity, unit_price, frequency, recency, monetary, country):
    df = pd.DataFrame([[
        quantity, unit_price, frequency, recency, monetary, country
    ]], columns=FEATURES)

    X_scaled = clv_scaler.transform(df)

    prediction = clv_model.predict(X_scaled)[0]

    return f"Estimated Customer Lifetime Value (CLV): ${prediction:.2f}"

inputs = [
    gr.Number(label="Quantity"),
    gr.Number(label="Unit Price"),
    gr.Number(label="Frequency"),
    gr.Number(label="Recency"),
    gr.Number(label="Monetary"),
    gr.Number(label="Country (Encoded)")
]

iface = gr.Interface(
    fn=predict_clv,
    inputs=inputs,
    outputs="text",
    title="Customer Lifetime Value Prediction",
    description="Predict expected CLV based on customer behavior and purchase patterns."
)

if __name__ == "__main__":
    iface.launch()
