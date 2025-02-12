from flask import Flask, jsonify
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load resources on startup
MODEL = None
SCALER = None
SEQ_SIZE = 60
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def load_resources():
    global MODEL, SCALER
    try:
        MODEL = load_model("models/2025_google_stock_price_lstm.model.keras")
        SCALER = joblib.load("models/2025_google_stock_price_scaler.gz")
        print("✅ Resources loaded successfully")
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        raise

@app.route('/predict/next_day', methods=['GET'])
def predict_next_day():
    """Returns prediction for the next trading day"""
    sequence = get_latest_sequence()
    prediction = MODEL.predict(sequence)
    
    # Fix inverse transformation (match training notebook lines 497-499)
    future_predictions_padded = np.concatenate((prediction, np.ones((1, 5))), axis=1)
    price = SCALER.inverse_transform(future_predictions_padded)[0][0]
    
    return jsonify({
        'prediction': price,
        'timestamp': (datetime.now() + timedelta(days=1)).isoformat()
    })

@app.route('/predict/next_week', methods=['GET'])
def predict_next_week():
    """Returns predictions for the next 7 trading days"""
    predictions = []
    sequence = get_latest_sequence().copy()
    
    for _ in range(7):
        # Predict next day
        pred = MODEL.predict(sequence)
        
        # Fix inverse transformation
        future_predictions_padded = np.concatenate((pred, np.ones((1, 5))), axis=1)
        price = SCALER.inverse_transform(future_predictions_padded)[0][0]
        
        # Store prediction
        predictions.append(price)
    
    return jsonify({
        'predictions': predictions,
        'start_date': (datetime.now() + timedelta(days=1)).isoformat(),
        'end_date': (datetime.now() + timedelta(days=7)).isoformat()
    })

def get_latest_sequence():
    """Retrieves and prepares the most recent sequence from latest data"""
    # Load all processed data (train + validate + test)
    train_df = pd.read_csv("data/processed/2025_google_stock_price_processed_train.csv")
    validate_df = pd.read_csv("data/processed/2025_google_stock_price_processed_validate.csv") 
    test_df = pd.read_csv("data/processed/2025_google_stock_price_processed_test.csv")
    
    # Combine and sort all data
    full_df = pd.concat([train_df, validate_df, test_df]).sort_values("Date")
    
    # Get last SEQ_SIZE samples from combined data
    df = full_df.tail(SEQ_SIZE)
    
    scaled_data = SCALER.transform(df[FEATURES])
    return scaled_data.reshape(1, SEQ_SIZE, len(FEATURES))

def create_synthetic_features(predicted_open, last_point):
    """Creates synthetic features for next time step"""
    new_point = last_point.copy()
    new_point[0] = predicted_open  # Open
    # Simple synthetic features - adjust these based on your training logic
    new_point[1] = predicted_open * 1.02  # High
    new_point[2] = predicted_open * 0.98  # Low
    new_point[3] = predicted_open * 1.01  # Close
    new_point[4] = new_point[3]  # Adj Close
    new_point[5] = last_point[5]  # Volume (keep previous)
    return new_point
if __name__ == '__main__':
    load_resources()
    app.run(host='0.0.0.0', port=8000, debug=False)