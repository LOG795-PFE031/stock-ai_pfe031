import datetime
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import pytz
import pandas_datareader as pdr
import math

# Update path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "model"))

# Constants - these might be updated based on available data
SCALER = joblib.load(os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_scaler.gz"))
POSSIBLE_FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_TARGET = "Open"  # Default target feature for prediction
SEQUENCE_LENGTH = 60     # Match with the PyTorch model

def scale_data(data, features):
    """Scale data using the pre-trained scaler with missing feature handling."""
    # Check for missing features in our scaler
    expected_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']  # Features the scaler was trained on
    
    # Detect missing features
    missing_features = [f for f in expected_features if f not in data.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        
        # Create a copy of the data and add the missing features with zeros
        # This ensures the scaler will work, even though predictions might be less accurate
        temp_data = data.copy()
        for feature in missing_features:
            # If we're missing 'Adj Close', use 'Close' values instead
            if feature == 'Adj Close' and 'Close' in data.columns:
                temp_data['Adj Close'] = data['Close']
                print("Using 'Close' values for 'Adj Close'")
            else:
                temp_data[feature] = 0.0
                print(f"Setting {feature} to zeros (predictions may be affected)")
        
        return SCALER.transform(temp_data[expected_features])
    
    # If all features are present, proceed normally
    return SCALER.transform(data[features])

def create_sequences(data, window_length=SEQUENCE_LENGTH):
    """Create sequences for LSTM models."""
    sequences = []
    for i in range(window_length, len(data)):
        seq = data[i-window_length:i]
        sequences.append(seq)
    return np.array(sequences)

def inverse_scale_predictions(predictions, features, target_idx):
    """Inverse transform predictions to original scale."""
    # Expected feature order in scaler
    expected_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Create a dummy array with the expected feature dimensions
    dummy = np.zeros((len(predictions), len(expected_features)))
    
    # Handle the case when target_feature doesn't match the scaler's order
    if target_idx == features.index('Open'):
        # Target is Open price, which is index 0 in expected_features
        target_scaler_idx = 0
    elif target_idx == features.index('Close'):
        # Target is Close price, which is index 3 in expected_features 
        target_scaler_idx = 3
    else:
        # Map the target index to the corresponding index in expected_features
        target_feature_name = features[target_idx]
        if target_feature_name in expected_features:
            target_scaler_idx = expected_features.index(target_feature_name)
        else:
            # Default to first feature if not found
            target_scaler_idx = 0
    
    # Place predictions in the right column of the dummy array
    dummy[:, target_scaler_idx] = predictions
    
    # Apply inverse transform
    scaled_back = SCALER.inverse_transform(dummy)
    
    # Return only the target feature column
    return scaled_back[:, target_scaler_idx]

def download_google_data():
    """Try multiple methods to download Google stock data."""
    # Method 1: Use pandas_datareader with different sources
    try:
        print("Trying to download data using pandas_datareader...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)  # 5 years
        
        # Try Stooq data source 
        df = pdr.data.DataReader('GOOGL', 'stooq', start_date, end_date)
        if not df.empty:
            print("Successfully downloaded data from Stooq")
            return df
    except Exception as e:
        print(f"pandas_datareader failed: {e}")
    
    # Method 2: Load from local backup if exists
    local_backup = os.path.join(PROJECT_ROOT, "data", "raw", "google_stock_backup.csv")
    if os.path.exists(local_backup):
        print(f"Loading from local backup: {local_backup}")
        df = pd.read_csv(local_backup, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    
    # Method 3: Use processed training data 
    processed_file = os.path.join(PROJECT_ROOT, "data", "processed", "2025_google_stock_price_processed_test.csv")
    if os.path.exists(processed_file):
        print(f"Loading from processed test dataset: {processed_file}")
        df = pd.read_csv(processed_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if not df.empty:
            return df
    
    # If all methods fail
    raise ValueError("Could not retrieve Google stock data from any source")

def calculate_metrics(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def plot_model_comparison(dates, actual, tf_pred, torch_pred, target_name):
    """Create and save comparison plots."""
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Plot 1: Full time range comparison
    plt.figure(figsize=(14, 8))
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, tf_pred, label='TensorFlow LSTM', linestyle='--')
    plt.plot(dates, torch_pred, label='PyTorch LSTM', linestyle=':')
    
    plt.title('Google Stock Price Prediction Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{target_name} Price (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "google_stock_price_predictions.png"))
    
    # Plot 2: Last 50 days for clearer visualization
    plt.figure(figsize=(14, 8))
    last_50_days = -50
    
    plt.plot(dates[last_50_days:], actual[last_50_days:], label='Actual', linewidth=2)
    plt.plot(dates[last_50_days:], tf_pred[last_50_days:], label='TensorFlow LSTM', linestyle='--')
    plt.plot(dates[last_50_days:], torch_pred[last_50_days:], label='PyTorch LSTM', linestyle=':')
    
    plt.title('Google Stock Price Prediction (Last 50 Days)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{target_name} Price (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "google_stock_price_predictions_last_50d.png"))
    
    # Plot 3: Predicted vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(actual, tf_pred, label='TensorFlow LSTM', alpha=0.5)
    plt.scatter(actual, torch_pred, label='PyTorch LSTM', alpha=0.5)
    
    # Add perfect prediction line (y=x)
    min_val = min(min(actual), min(tf_pred), min(torch_pred))
    max_val = max(max(actual), max(tf_pred), max(torch_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    plt.title('Predicted vs Actual Stock Prices', fontsize=16)
    plt.xlabel('Actual Price (USD)', fontsize=14)
    plt.ylabel('Predicted Price (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "comparison_2025_stock_predictions.png"))
    
    return

if __name__ == "__main__":
    # Data loading and preprocessing with robust error handling
    try:
        # First try yfinance
        try:
            goog = yf.Ticker("GOOG")
            price_data = goog.history(period='5y')
            
            if price_data.empty:
                print("No data found for GOOG ticker, trying GOOGL...")
                goog = yf.Ticker("GOOGL")
                price_data = goog.history(period='5y')
                
            if price_data.empty:
                raise ValueError("yfinance API failed to retrieve data")
        except Exception as e:
            print(f"yfinance API error: {e}")
            # Fall back to alternative method
            price_data = download_google_data()
            
        print(f"Successfully loaded {len(price_data)} days of Google stock price data")
        print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        
    except Exception as e:
        print(f"Fatal error loading data: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Install pandas-datareader: pip install pandas-datareader")
        print("3. Create a data/raw directory with a google_stock_backup.csv file")
        print("4. Make sure you've run preprocessing to generate the processed test dataset")
        sys.exit(1)
    
    # Clean data
    price_data = price_data.dropna()
    
    # Check for expected features and adapt if necessary
    print("Available columns:", price_data.columns.tolist())
    
    # Find which of our possible features are available in the data
    available_features = [f for f in POSSIBLE_FEATURES if f in price_data.columns]
    
    if len(available_features) < 4:  # We need at least some features
        raise ValueError("Not enough valid features found in the data")
    
    print(f"Using features: {available_features}")
    
    # Determine the target feature based on availability
    target_feature = DEFAULT_TARGET if DEFAULT_TARGET in price_data.columns else available_features[0]
    target_idx = available_features.index(target_feature)
    print(f"Using '{target_feature}' as target feature for prediction")
    
    # Add pct_change if it doesn't exist
    if 'pct_change' not in price_data.columns and 'Close' in price_data.columns:
        price_data['pct_change'] = price_data['Close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Check if data is in reverse order (Stooq data comes in descending order)
    if price_data.index[0] > price_data.index[-1]:
        print("Data is in reverse order - sorting chronologically...")
        price_data = price_data.sort_index()
        print(f"New date range: {price_data.index[0]} to {price_data.index[-1]}")

    # Train/test split - Use tz-naive datetime for comparison
    training_cutoff = datetime.datetime(2022, 1, 1)  # No timezone info

    # For safety, convert index to tz-naive if it has timezone info
    if price_data.index.tz is not None:
        price_data.index = price_data.index.tz_localize(None)

    # Select test data
    test_data = price_data[price_data.index >= training_cutoff]

    # Handle case if test data is empty
    if len(test_data) == 0:
        print("Warning: No test data found after 2022-01-01. Using last 20% of data.")
        split_idx = int(len(price_data) * 0.8)
        test_data = price_data.iloc[split_idx:]
        
    print(f"Using {len(test_data)} days for testing")
    
    # Feature scaling
    scaled_data = scale_data(price_data, available_features)
    scaled_test_data = scaled_data[-len(test_data)-SEQUENCE_LENGTH:]  # Ensure enough history for sequences

    # Create testing sequences for both models
    print(f"Creating sequences with shape: {scaled_test_data.shape}")
    
    # TensorFlow LSTM expects (samples, timesteps, features)
    tf_test_sequences = create_sequences(scaled_test_data, SEQUENCE_LENGTH)
    
    # Determine the correct reshape dimensions
    print(f"Sequence shape before reshape: {tf_test_sequences.shape}")
    num_sequences = tf_test_sequences.shape[0]
    num_features = scaled_test_data.shape[1]
    
    # Make sure the reshape is mathematically valid
    # Shape should be (num_sequences, sequence_length, num_features)
    tf_test_sequences = tf_test_sequences.reshape(num_sequences, SEQUENCE_LENGTH, num_features)
    print(f"Sequence shape after reshape: {tf_test_sequences.shape}")
    
    # PyTorch LSTM expects only the target feature (samples, timesteps, 1)
    torch_test_sequences = torch.tensor(
        tf_test_sequences[..., target_idx].reshape(-1, SEQUENCE_LENGTH, 1), 
        dtype=torch.float32
    )
    
    # These are the actual values we're trying to predict
    # Note: We need to be careful about offsets - the SEQUENCE_LENGTH affects how many predictions we can make
    actual_values = test_data[target_feature].values
    
    # Print length for debugging
    print(f"Length of test data: {len(test_data)}")
    print(f"Length of actual values before offset: {len(actual_values)}")
    
    # Load models with error handling
    try:
        tf_model_path = os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_lstm.model.keras")
        torch_model_path = os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth")
        
        # Check if model files exist
        if not os.path.exists(tf_model_path):
            raise FileNotFoundError(f"TensorFlow model not found at {tf_model_path}")
        if not os.path.exists(torch_model_path):
            raise FileNotFoundError(f"PyTorch model not found at {torch_model_path}")
            
        # Load models
        tf_model = load_model(tf_model_path)
        torch_model = torch.load(torch_model_path)
        torch_model.eval()  # Set to evaluation mode
        
        print("Successfully loaded both models")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Generate predictions from TensorFlow model
    tf_predictions = tf_model.predict(tf_test_sequences)
    tf_predictions = tf_predictions.flatten()
    tf_predictions_unscaled = inverse_scale_predictions(tf_predictions, available_features, target_idx)
    
    # Generate predictions from PyTorch model
    with torch.no_grad():
        torch_predictions = torch_model(torch_test_sequences).numpy().flatten()
    torch_predictions_unscaled = inverse_scale_predictions(torch_predictions, available_features, target_idx)
    
    # Print lengths for debugging
    print(f"Length of TensorFlow predictions: {len(tf_predictions_unscaled)}")
    print(f"Length of PyTorch predictions: {len(torch_predictions_unscaled)}")
    
    # Adjust target values to match prediction length
    # For each sequence, we make one prediction, which means we lose SEQUENCE_LENGTH time steps from the beginning
    actual_values_aligned = actual_values[SEQUENCE_LENGTH:]
    
    # Ensure all arrays have the same length (use the minimum length to be safe)
    min_length = min(len(actual_values_aligned), len(tf_predictions_unscaled), len(torch_predictions_unscaled))
    print(f"Minimum common length: {min_length}")
    
    actual_values_aligned = actual_values_aligned[:min_length]
    tf_predictions_unscaled = tf_predictions_unscaled[:min_length]
    torch_predictions_unscaled = torch_predictions_unscaled[:min_length]
    
    # Calculate metrics for both models
    tf_metrics = calculate_metrics(actual_values_aligned, tf_predictions_unscaled)
    torch_metrics = calculate_metrics(actual_values_aligned, torch_predictions_unscaled)
    
    # Print the metrics for comparison
    print("\n========== MODEL PERFORMANCE COMPARISON ==========")
    print("Metric\t\tTensorFlow LSTM\tPyTorch LSTM")
    print("-" * 50)
    for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
        print(f"{metric}\t\t{tf_metrics[metric]:.4f}\t\t{torch_metrics[metric]:.4f}")
    
    # Create a DataFrame to easily display and compare results
    results_df = pd.DataFrame({
        'TensorFlow LSTM': tf_predictions_unscaled,
        'PyTorch LSTM': torch_predictions_unscaled,
        'Actual': actual_values_aligned
    })
    
    # Print a sample of the predictions
    print("\n========== SAMPLE PREDICTIONS ==========")
    print(results_df.head(10))
    
    # Get aligned dates for plotting
    # Since we're predicting from sequence_length onwards and may have further trimmed our data,
    # we need to align the dates as well
    plot_dates = test_data.index[SEQUENCE_LENGTH:SEQUENCE_LENGTH + min_length]
    
    # Plot the results
    plot_model_comparison(
        plot_dates,
        actual_values_aligned, 
        tf_predictions_unscaled, 
        torch_predictions_unscaled,
        target_feature
    )
    
    print("\nAnalysis complete! Visualizations saved to reports directory.")
    plt.show()