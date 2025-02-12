import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Add these constants (update path if different)
SCALER = joblib.load("../models/2025_google_stock_price_scaler.gz")
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
SEQ_SIZE = 60
def load_recent_data(days=60):
    """Loads the most recent data needed for predictions"""
    df = pd.read_csv("../data/processed/2025_google_stock_price_processed_test.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").tail(days)

def prepare_sequence(data):
    """Prepares input sequence for model prediction"""
    scaled_data = SCALER.transform(data[FEATURES])
    sequence = scaled_data[-SEQ_SIZE:].reshape(1, SEQ_SIZE, -1)
    return sequence