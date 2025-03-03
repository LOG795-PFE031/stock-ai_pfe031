import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import json
from celery_tasks import process_prediction, analyze_prediction
import yfinance as yf
import pandas_datareader as pdr
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "model"))

# Constants
SCALER = joblib.load(os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_scaler.gz"))
EXPECTED_FEATURES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
DEFAULT_TARGET = "Open"  # We're predicting the 'Open' price
SEQUENCE_LENGTH = 60  # Match with the model

# Celery task queues
QUEUE_NAMES = ['prediction_queue_1', 'prediction_queue_2']  # Multiple queues for processing

def get_latest_stock_data(days=120):  # Make sure we get enough data, use a larger window to be safe
    """Fetch the latest stock data for Google."""
    try:
        # Try yfinance first
        ticker = "GOOGL"  # Google ticker
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"yfinance download failed: {e}")
        
        # Try pandas_datareader as fallback
        try:
            data = pdr.data.DataReader(ticker, 'stooq', start_date, end_date)
            if not data.empty:
                logger.info("Successfully fetched data from Stooq")
                # If data is in reverse order (newer dates first), sort it
                if data.index[0] > data.index[-1]:
                    data = data.sort_index()
                return data
        except Exception as e:
            logger.warning(f"pandas_datareader failed: {e}")
        
        # Final fallback: use historical data
        processed_file = os.path.join(PROJECT_ROOT, "data", "processed", "2025_google_stock_price_processed_test.csv")
        if os.path.exists(processed_file):
            logger.info(f"Using historical data from: {processed_file}")
            df = pd.read_csv(processed_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df.tail(days)
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise
    raise ValueError("Could not retrieve stock data from any source")

def preprocess_data(data):
    """Preprocess data for prediction."""
    # Make sure we have enough data
    if len(data) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data points. Need at least {SEQUENCE_LENGTH}, but got {len(data)}.")
    
    logger.info(f"Data shape before preprocessing: {data.shape}")
    
    # Handle missing features
    missing_features = [f for f in EXPECTED_FEATURES if f not in data.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        
        # Create a copy and add missing features
        temp_data = data.copy()
        for feature in missing_features:
            if feature == 'Adj Close' and 'Close' in data.columns:
                temp_data['Adj Close'] = data['Close']
                logger.info("Using 'Close' values for 'Adj Close'")
            else:
                temp_data[feature] = 0.0
                logger.info(f"Setting {feature} to zeros")
        
        processed_data = temp_data[EXPECTED_FEATURES]
    else:
        processed_data = data[EXPECTED_FEATURES]
    
    # Scale the data
    scaled_data = SCALER.transform(processed_data)
    logger.info(f"Scaled data shape: {scaled_data.shape}")
    
    # If we have more data than needed, just take the most recent SEQUENCE_LENGTH points
    if len(scaled_data) > SEQUENCE_LENGTH:
        sequence_data = scaled_data[-SEQUENCE_LENGTH:]
    else:
        sequence_data = scaled_data
    
    # Create sequence for the model (shape: [1, SEQUENCE_LENGTH, num_features])
    sequence = sequence_data.reshape(1, SEQUENCE_LENGTH, len(EXPECTED_FEATURES))
    logger.info(f"Sequence shape for model: {sequence.shape}")
    
    # Get the index of the target feature
    if DEFAULT_TARGET in EXPECTED_FEATURES:
        target_idx = EXPECTED_FEATURES.index(DEFAULT_TARGET)
    else:
        target_idx = 0  # Default to first feature
    
    # Extract only the target feature for PyTorch model (shape: [1, SEQUENCE_LENGTH, 1])
    pytorch_sequence = torch.tensor(
        sequence[..., target_idx].reshape(1, SEQUENCE_LENGTH, 1),
        dtype=torch.float32
    )
    
    return pytorch_sequence, target_idx

def predict_next_day(data, model_path):
    """Predict the next day's opening price."""
    # Load the model
    model = torch.load(model_path)
    model.eval()  # Set to evaluation mode
    
    # Preprocess the data
    sequence, target_idx = preprocess_data(data)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(sequence).numpy().flatten()
    
    # Inverse transform the prediction
    dummy = np.zeros((len(prediction), len(EXPECTED_FEATURES)))
    dummy[:, target_idx] = prediction
    prediction_unscaled = SCALER.inverse_transform(dummy)[:, target_idx][0]
    
    return prediction_unscaled

def send_to_celery_tasks(prediction, timestamp):
    """Send prediction to Celery tasks for processing."""
    # Create the message payload
    message = {
        'ticker': 'GOOGL',
        'prediction_date': timestamp.strftime('%Y-%m-%d'),
        'prediction_time': timestamp.strftime('%H:%M:%S'),
        'predicted_open': float(prediction),
        'model': 'pytorch_lstm'
    }
    
    # Log the prediction
    logger.info(f"Sending prediction to Celery tasks: ${prediction:.2f} for {timestamp.strftime('%Y-%m-%d')}")
    
    try:
        # Send to both task queues asynchronously
        # Task 1: Basic processing
        process_task = process_prediction.delay(message)
        
        # Task 2: Advanced analysis
        analyze_task = analyze_prediction.delay(message)
        
        logger.info(f"Successfully sent to Celery tasks:")
        logger.info(f"  Process task ID: {process_task.id}")
        logger.info(f"  Analysis task ID: {analyze_task.id}")
        
        # Return the task IDs if you want to track them
        return {
            'process_task_id': process_task.id,
            'analyze_task_id': analyze_task.id
        }
        
    except Exception as e:
        logger.error(f"Error sending to Celery tasks: {e}")
        
        # Fallback to direct execution (for demo/testing)
        logger.warning("Falling back to direct execution...")
        
        # Execute the tasks directly (synchronously)
        try:
            # Process the prediction directly
            process_result = process_prediction(message)
            analyze_result = analyze_prediction(message)
            
            logger.info("Direct execution results:")
            logger.info(f"  Process result: {process_result}")
            logger.info(f"  Analysis result: {analyze_result}")
            
            return {
                'process_result': process_result,
                'analyze_result': analyze_result
            }
        except Exception as direct_error:
            logger.error(f"Direct execution failed: {direct_error}")
            return None

def run_prediction_service():
    """Main function to fetch data, make a prediction, and publish it."""
    try:
        # Get the latest stock data
        data = get_latest_stock_data()
        logger.info(f"Retrieved {len(data)} days of stock data")
        
        # Load model and predict
        model_path = os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth")
        next_day_prediction = predict_next_day(data, model_path)
        
        # Calculate the next trading day (simple approximation)
        last_date = data.index[-1]
        # If today is Friday, next trading day is Monday (+3 days)
        if last_date.weekday() == 4:  # Friday
            next_date = last_date + datetime.timedelta(days=3)
        # If today is Saturday, next trading day is Monday (+2 days)
        elif last_date.weekday() == 5:  # Saturday
            next_date = last_date + datetime.timedelta(days=2)
        # If today is Sunday, next trading day is Monday (+1 day)
        elif last_date.weekday() == 6:  # Sunday
            next_date = last_date + datetime.timedelta(days=1)
        else:
            next_date = last_date + datetime.timedelta(days=1)
        
        # Format the prediction
        logger.info(f"Predicted opening price for {next_date.strftime('%Y-%m-%d')}: ${next_day_prediction:.2f}")
        
        # Send the prediction to Celery tasks
        task_results = send_to_celery_tasks(next_day_prediction, next_date)
        
        return {
            'date': next_date.strftime('%Y-%m-%d'),
            'prediction': next_day_prediction
        }
    
    except Exception as e:
        logger.error(f"Prediction service failed: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting prediction service")
        prediction = run_prediction_service()
        logger.info(f"Prediction service completed successfully")
        logger.info(f"Predicted ${prediction['prediction']:.2f} for {prediction['date']}")
    except Exception as e:
        logger.error(f"Error in prediction service: {e}")
        sys.exit(1)