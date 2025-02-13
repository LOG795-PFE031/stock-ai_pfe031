import datetime
import pandas as pd
import torch
import time
import numpy as np
import joblib
import os
import sys
import tensorflow as tf
from matplotlib import dates as mdates

# Update path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Get current directory of script
sys.path.append(os.path.join(PROJECT_ROOT, "model"))  # Explicitly add model package

import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model.helpers import train, predict
from model.preprocessing import process_inputs, process_targets

# Update scaler path (line 17)
SCALER = joblib.load(os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_scaler.gz"))

# After SCALER definition (line 17)
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# For PyTorch compatibility:
def scale_for_torch(data):
    """Align with TensorFlow's preprocessing"""
    return SCALER.transform(data[FEATURES])

# Add to imports (line 15)
from tensorflow.keras.models import load_model

print(tf.__version__)  # Should show 2.x.x

# Move this function ABOVE the __main__ block
def create_rolling_sequences(data, window_length=63):
    """Modified version of TF's construct_lstm_data for percentage returns"""
    sequences = []
    for i in range(window_length, len(data)):
        seq = data['pct_change'][i-window_length:i].values
        sequences.append(seq)
    return np.array(sequences).reshape(-1, window_length, 1)

def inverse_scale_predictions(predictions):
    """Properly inverse transform predictions using the original scaler"""
    # Reshape predictions to 1D array
    predictions = np.array(predictions).squeeze()  # Removes singleton dimensions
    
    # Create dummy array with same shape as training data (6 features)
    dummy = np.zeros((len(predictions), 6))
    dummy[:, 0] = predictions  # Now compatible with 1D predictions
    return SCALER.inverse_transform(dummy)[:, 0]  # Return only Open prices

def convert_returns_to_price(predictions, data):
    """Convert percentage return predictions to price values"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy().squeeze()
    else:
        predictions = np.array(predictions).squeeze()
    
    # Clip extreme returns to ±50% daily
    predictions = np.clip(predictions, -0.5, 0.5)
    
    base_prices = data['Close'].values[-len(predictions):]
    return base_prices * (1 + predictions)

def calculate_metrics(actual, predicted):
    """Calculate common regression metrics with length alignment"""
    min_length = min(len(actual), len(predicted))
    actual = actual[-min_length:]
    predicted = predicted[-min_length:]
    
    mask = ~np.isnan(predicted)
    if mask.sum() == 0:
        return "All predictions NaN"
    return f"R²: {r2_score(actual[mask], predicted[mask]):.4f}, MAE: {mean_absolute_error(actual[mask], predicted[mask]):.4f}"

def add_common_visuals():
    """Reuse formatting from notebook 3-model-training"""
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.ylabel("Stock Price (USD)")

def plot_combined_results(tf_preds, torch_preds, actual):
    """Visual comparison of both models' predictions"""
    plt.figure(figsize=(12,6))
    plt.plot(actual.values, label='Actual Prices', color='black')
    plt.plot(tf_preds, label='TensorFlow Predictions', linestyle='--')
    plt.plot(torch_preds, label='PyTorch Predictions', linestyle=':')
    add_common_visuals()
    plt.legend()
    plt.title("Model Comparison: TensorFlow vs PyTorch Predictions")

if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    goog = yf.Ticker("GOOG")  # Get Google stock data
    price_series = goog.history(period='max')['Close'].dropna()

    perf_series = price_series.pct_change().dropna()

    # More robust configuration
    training_cutoff = datetime.datetime(2022, 1, 1, tzinfo=pytz.timezone('America/New_York'))  # Real training cutoff
    x_df = process_inputs(perf_series, window_length=63)  # 3 months of trading days (~quarterly context)
    y_series = process_targets(perf_series)

    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    # Isolate training data
    training_x_series = x_df.loc[x_df.index < training_cutoff]
    training_y_series = y_series.loc[y_series.index < training_cutoff]

    # Move this section up before training
    # 1. Load TensorFlow model first
    model_path = os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_lstm.model.keras")
    tf_model = load_model(model_path)
    
    # 2. Train PyTorch model
   # trained_model = train(training_x_series, training_y_series)
    #torch.save(trained_model, os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth"))

    # 3. Load PyTorch model for comparison
    torch_model = torch.load(os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth"))

    # Isolate test data
    test_x_series = x_df.loc[x_df.index >= training_cutoff]
    actual_series = y_series.loc[y_series.index >= training_cutoff]

    forecast_series = predict(torch_model, test_x_series)
    results_df = forecast_series.to_frame('Forecast').join(actual_series.to_frame('Actual')).dropna()

    # Convert returns to dollar values
    results_df = results_df.join(price_series.rename('RefPrice'))
    results_df['PredictedPrice'] = results_df['RefPrice'] * (1 + results_df['Forecast'])
    results_df['ActualPrice'] = results_df['RefPrice'] * (1 + results_df['Actual'])

    # Verify we have actual data for comparison
    max_actual_date = price_series.index.max()
    print(f"\nLatest actual data date: {max_actual_date.strftime('%Y-%m-%d')}")
    print(f"Model test period: {test_x_series.index.min().strftime('%Y-%m-%d')} to {test_x_series.index.max().strftime('%Y-%m-%d')}")

    # Comment out or remove these visualization blocks:
    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()
    
    print("\nDetailed Price Predictions:")
    for date, row in results_df.iterrows():
        target_date = date + pd.Timedelta(days=2)
        print(f"{date.strftime('%Y-%m-%d')} prediction -> {target_date.strftime('%Y-%m-%d')}:")
        print(f"  Predicted: ${row['PredictedPrice']:.2f}")
        print(f"  Actual:    ${row['ActualPrice']:.2f}\n")

    # 1. Use common test data from preprocessing
    test_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "2025_google_stock_price_processed_test.csv")
    test_data = pd.read_csv(test_data_path)

    # Modify the percentage change calculation to avoid division by zero
    test_data['pct_change'] = test_data['Close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Add NaN filtering in sequence generation
    test_sequences = create_rolling_sequences(test_data.dropna(), window_length=63)

    # Modify the TensorFlow sequence generation
    test_data_scaled = scale_for_torch(test_data)  # Use all 6 features
    test_sequences_tf = np.stack([test_data_scaled[i-63:i] for i in range(63, len(test_data_scaled))])
    test_sequences_tf = test_sequences_tf.reshape(-1, 63, len(FEATURES))  # 6 features

    # Then keep existing prediction code:
    test_predictions_tf = tf_model.predict(test_sequences_tf, verbose=0)

    # 3. PyTorch inference (needs adaptation)
    test_predictions_torch = torch_model(torch.Tensor(test_sequences).squeeze(-1)).detach()
    test_predictions_torch_prices = inverse_scale_predictions(test_predictions_torch.numpy())

    # Calculate actual prices from test_data
    actual_prices = test_data['Open'].values[63:]  # Use original Open prices

    # Convert predictions to prices
    test_predictions_tf_prices = inverse_scale_predictions(test_predictions_tf)
    test_predictions_torch_prices = test_predictions_torch_prices[-len(actual_prices):]

    # Find common prediction length
    min_length = min(len(test_predictions_tf_prices), len(test_predictions_torch_prices), len(actual_prices))

    # Slice all arrays to match
    actual_prices = actual_prices[-min_length:]
    test_predictions_tf_prices = test_predictions_tf_prices[-min_length:]
    test_predictions_torch_prices = test_predictions_torch_prices[-min_length:]

    # Now compare metrics
    print(f"{'R² Score':<15} {r2_score(actual_prices, test_predictions_tf_prices):<12.4f} {r2_score(actual_prices, test_predictions_torch_prices):<12.4f}")
    print(f"{'MAE':<15} {mean_absolute_error(actual_prices, test_predictions_tf_prices):<12.4f} {mean_absolute_error(actual_prices, test_predictions_torch_prices):<12.4f}")

    # Generate comparison plot using existing data
    plot_combined_results(
        tf_preds=test_predictions_tf,
        torch_preds=test_predictions_torch,
        actual=test_data['Open']
    )

    # Add directory creation before saving plot
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)  # Create if not exists

    plt.savefig(os.path.join(report_dir, "comparison_2025_stock_predictions.png"))

    # Reuse API Server's sequence generation from utils/data_loader.py
    test_sequences = create_rolling_sequences(test_data, window_length=63)

    # Benchmark prediction latency
    def benchmark_inference(model, iterations=100):
        """Measure median prediction latency for TensorFlow model"""
        # Generate proper input sequence matching training specs
        test_seq = test_sequences_tf[:1]  # Use actual TF input shape (1, 63, 6)
        
        times = []
        for _ in range(iterations):
            start = time.time()
            model.predict(test_seq, verbose=0)
            times.append(time.time() - start)
        return np.median(times)

    # For PyTorch benchmarking (keep existing):
    def benchmark_inference_torch(model, iterations=100):
        """Measure median prediction latency for PyTorch model"""
        test_seq = torch.Tensor(test_sequences[:1]).squeeze(-1)
        times = []
        for _ in range(iterations):
            start = time.time()
            with torch.no_grad():
                model(test_seq)
            times.append(time.time() - start)
        return np.median(times)

    tf_latency = benchmark_inference(tf_model)
    torch_latency = benchmark_inference_torch(torch_model)

    print(f"TensorFlow Latency: {tf_latency:.4f}s")
    print(f"PyTorch Latency: {torch_latency:.4f}s")

    print(f"Generated sequences: {test_sequences.shape[0]} (needs >=1)")

    print("PyTorch input shape:", test_sequences.shape)  # Should be (N, 63, 1)
    print("After squeeze:", torch.Tensor(test_sequences).squeeze(-1).shape)  # Should be (N, 63)

    # Check for remaining NaNs
    print("Remaining NaNs in data:", test_data.isna().sum().sum())
    print("Valid predictions count:", len(test_predictions_torch_prices[~np.isnan(test_predictions_torch_prices)]))

    # Replace existing metric printing with:
    print("\nModel Performance Summary:")
    print(f"{'Metric':<15} {'TensorFlow':<12} {'PyTorch':<12}")
    print(f"{'R² Score':<15} {r2_score(actual_prices, test_predictions_tf_prices):<12.4f} {r2_score(actual_prices, test_predictions_torch_prices):<12.4f}")
    print(f"{'MAE':<15} {mean_absolute_error(actual_prices, test_predictions_tf_prices):<12.4f} {mean_absolute_error(actual_prices, test_predictions_torch_prices):<12.4f}")
    print(f"{'Latency (ms)':<15} {tf_latency*1000:<12.2f} {torch_latency*1000:<12.2f}")

    # Get the number of available predictions
    pred_count = len(actual_prices)
    display_count = min(3, pred_count)  # Show up to 3 latest

    print(f"\nLast {display_count} Predictions Comparison:")
    print(f"{'Date':<12} {'Actual':<8} {'TF Pred':<8} {'Torch Pred':<8}")

    if pred_count == 0:
        print("No predictions available")
    else:
        dates = test_data['Date'].iloc[-pred_count:].tolist()  # Convert to regular list
        for i in range(-display_count, 0):
            idx = i if i != -0 else 0  # Handle -0 edge case
            print(f"{dates[idx]:<12} ${actual_prices[idx]:<7.2f} ${test_predictions_tf_prices[idx]:<7.2f} ${test_predictions_torch_prices[idx]:<7.2f}")
