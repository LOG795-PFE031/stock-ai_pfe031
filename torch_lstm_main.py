import datetime
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
import pytz

# Update path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "model"))

from model.helpers import train, predict
from model.preprocessing import process_inputs, process_targets

# Constants
SCALER = joblib.load(os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_scaler.gz"))
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Open"  # Assuming we're predicting the 'Open' price

def scale_data(data):
    """Scale data using the pre-trained scaler."""
    return SCALER.transform(data[FEATURES])

def create_sequences(data, window_length=63):
    """Create sequences from the scaled data."""
    sequences = []
    for i in range(window_length, len(data)):
        sequences.append(data[i-window_length:i])
    return np.array(sequences)

def inverse_scale_predictions(predictions):
    """Inverse transform predictions using the original scaler."""
    dummy = np.zeros((len(predictions), len(FEATURES)))
    dummy[:, FEATURES.index(TARGET_FEATURE)] = predictions.squeeze()
    return SCALER.inverse_transform(dummy)[:, FEATURES.index(TARGET_FEATURE)]

def convert_returns_to_price(predictions, data):
    """Convert percentage returns to price values using shifted Close prices."""
    predictions = np.array(predictions).squeeze()
    predictions = np.clip(predictions, -0.5, 0.5)
    base_prices = data['Close'].shift(1).values[-len(predictions):]
    return base_prices * (1 + predictions)

def calculate_metrics(actual, predicted):
    """Calculate regression metrics with alignment and NaN handling."""
    min_length = min(len(actual), len(predicted))
    actual = actual[-min_length:]
    predicted = predicted[-min_length:]
    mask = ~np.isnan(predicted)
    if mask.sum() == 0:
        return "All predictions NaN"
    return f"R²: {r2_score(actual[mask], predicted[mask]):.4f}, MAE: {mean_absolute_error(actual[mask], predicted[mask]):.4f}"

def add_common_visuals():
    """Standard visualization formatting."""
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.ylabel("Stock Price (USD)")

def plot_combined_results(tf_preds, torch_preds, actual, dates):
    """Plot comparison of model predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Prices', color='black')
    plt.plot(dates, tf_preds, label='TensorFlow Predictions', linestyle='--')
    plt.plot(dates, torch_preds, label='PyTorch Predictions', linestyle=':')
    add_common_visuals()
    plt.legend()
    plt.title("Model Comparison: TensorFlow vs PyTorch Predictions")

def benchmark_inference(model, input_data, iterations=100):
    """Benchmark model inference latency."""
    times = []
    for _ in range(iterations):
        start = time.time()
        if isinstance(model, tf.keras.Model):
            model.predict(input_data[:1], verbose=0)
        else:
            with torch.no_grad():
                model(torch.Tensor(input_data[:1]))
        times.append(time.time() - start)
    return np.median(times)

if __name__ == "__main__":
    # Data loading and preprocessing
    goog = yf.Ticker("GOOG")
    price_data = goog.history(period='max').dropna()
    price_data['pct_change'] = price_data['Close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Train/test split
    training_cutoff = datetime.datetime(2022, 1, 1, tzinfo=pytz.timezone('America/New_York'))
    test_data = price_data[price_data.index >= training_cutoff]
    
    # Feature scaling
    scaled_data = scale_data(price_data)
    scaled_test_data = scaled_data[-len(test_data)-63:]  # Ensure enough history for sequences

    # Sequence generation
    test_sequences_tf = create_sequences(scaled_test_data, 63)
    test_sequences_tf = test_sequences_tf.reshape(-1, 63, len(FEATURES))

    # Load models
    tf_model = load_model(os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_lstm.model.keras"))
    torch_model = torch.load(os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth"))

    # Generate predictions
    tf_predictions = tf_model.predict(test_sequences_tf, verbose=0).squeeze()
    tf_prices = inverse_scale_predictions(tf_predictions)

    torch_input = torch.Tensor(test_sequences_tf[..., FEATURES.index(TARGET_FEATURE)].reshape(-1, 63, 1))
    torch_predictions = torch_model(torch_input).detach().numpy().squeeze()
    torch_prices = inverse_scale_predictions(torch_predictions)

    # Align data
    actual_prices = test_data[TARGET_FEATURE].values[63:]
    min_length = min(len(tf_prices), len(torch_prices), len(actual_prices))
    dates = test_data.index[63:63+min_length]

    # Calculate metrics
    print("\nModel Performance Summary:")
    print(f"{'Metric':<15} {'TensorFlow':<12} {'PyTorch':<12}")
    print(f"{'R² Score':<15} {r2_score(actual_prices[:min_length], tf_prices[:min_length]):.4f} {r2_score(actual_prices[:min_length], torch_prices[:min_length]):.4f}")
    print(f"{'MAE':<15} {mean_absolute_error(actual_prices[:min_length], tf_prices[:min_length]):.4f} {mean_absolute_error(actual_prices[:min_length], torch_prices[:min_length]):.4f}")

    # Benchmark latency
    tf_latency = benchmark_inference(tf_model, test_sequences_tf)
    torch_latency = benchmark_inference(torch_model, test_sequences_tf[..., FEATURES.index(TARGET_FEATURE)])
    print(f"\nLatency (median):\nTensorFlow: {tf_latency*1000:.2f}ms\nPyTorch: {torch_latency*1000:.2f}ms")

    # Plot results
    plot_combined_results(
        tf_prices[:min_length],
        torch_prices[:min_length],
        actual_prices[:min_length],
        dates[:min_length]
    )
    
    # Save report
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(os.path.join(report_dir, "comparison_2025_stock_predictions.png"))
    plt.show()

    # Print sample predictions
    print("\nLatest Predictions:")
    for i in range(1, 4):
        if i > min_length:
            break
        idx = -i
        print(f"{dates[idx].strftime('%Y-%m-%d')}:")
        print(f"  Actual: ${actual_prices[idx]:.2f}")
        print(f"  TF Pred: ${tf_prices[idx]:.2f} | Torch Pred: ${torch_prices[idx]:.2f}\n")