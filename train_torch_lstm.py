import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add model directory to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "model"))
from model.ltsm_model import LSTMStocksModule

# Constants
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Open"
SEQUENCE_SIZE = 60
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001

def load_data():
    """Load preprocessed data splits."""
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    
    train_df = pd.read_csv(os.path.join(data_dir, "2025_google_stock_price_processed_train.csv"))
    validate_df = pd.read_csv(os.path.join(data_dir, "2025_google_stock_price_processed_validate.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "2025_google_stock_price_processed_test.csv"))
    
    # Convert dates
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    validate_df["Date"] = pd.to_datetime(validate_df["Date"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    
    return train_df, validate_df, test_df

def create_sequences(data, seq_length=SEQUENCE_SIZE):
    """Create sequences for LSTM input."""
    feature_data = data[FEATURES].values
    X, y = [], []
    
    for i in range(seq_length, len(feature_data)):
        X.append(feature_data[i-seq_length:i])
        y.append(feature_data[i, FEATURES.index(TARGET_FEATURE)])
    
    return np.array(X), np.array(y)

def prepare_torch_data(X, y):
    """Convert numpy arrays to PyTorch tensors."""
    # Extract only the TARGET_FEATURE column (Open price) for sequences
    X_tensor = torch.tensor(X[..., FEATURES.index(TARGET_FEATURE)].reshape(-1, SEQUENCE_SIZE, 1), dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    
    return TensorDataset(X_tensor, y_tensor)

def train_model():
    """Train PyTorch LSTM model."""
    # Load data
    train_df, validate_df, test_df = load_data()
    
    # Load scaler
    scaler_path = os.path.join(PROJECT_ROOT, "models", "2025_google_stock_price_scaler.gz")
    scaler = joblib.load(scaler_path)
    
    # Create sequences
    X_train, y_train = create_sequences(train_df)
    X_val, y_val = create_sequences(validate_df)
    X_test, y_test = create_sequences(test_df)
    
    # Prepare data for PyTorch
    train_dataset = prepare_torch_data(X_train, y_train)
    val_dataset = prepare_torch_data(X_val, y_val)
    test_dataset = prepare_torch_data(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LSTMStocksModule()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(PROJECT_ROOT, "models", "torch_lstm_model.pth"))
            
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('PyTorch LSTM Model Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    plt.savefig(os.path.join(PROJECT_ROOT, "reports", "torch_lstm_training.png"))
    plt.show()
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test[..., FEATURES.index(TARGET_FEATURE)].reshape(-1, SEQUENCE_SIZE, 1), dtype=torch.float32)
        y_pred = model(X_test_tensor).numpy()
        
    # Convert predictions back to original scale
    dummy = np.zeros((len(y_pred), len(FEATURES)))
    dummy[:, FEATURES.index(TARGET_FEATURE)] = y_pred.squeeze()
    y_pred_inv = scaler.inverse_transform(dummy)[:, FEATURES.index(TARGET_FEATURE)]
    
    # Convert actual values back to original scale
    dummy = np.zeros((len(y_test), len(FEATURES)))
    dummy[:, FEATURES.index(TARGET_FEATURE)] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, FEATURES.index(TARGET_FEATURE)]
    
    # Plot test predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Date'].values[SEQUENCE_SIZE:], y_test_inv, label='Actual', color='blue')
    plt.plot(test_df['Date'].values[SEQUENCE_SIZE:], y_pred_inv, label='Predicted', color='orange')
    plt.title('Google Stock Price Prediction (PyTorch LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "reports", "torch_lstm_predictions.png"))
    plt.show()
    
    print(f"PyTorch model saved to: {os.path.join(PROJECT_ROOT, 'models', 'torch_lstm_model.pth')}")
    return model

if __name__ == "__main__":
    train_model()