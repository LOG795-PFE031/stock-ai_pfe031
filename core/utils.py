"""
Utility functions for the Stock AI system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Calculate Returns (percentage change)
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate Volatility (20-day standard deviation of returns)
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Ensure Adj Close exists (if not, use Close)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate a stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-5 characters, alphanumeric
    return bool(symbol and 1 <= len(symbol) <= 5 and symbol.isalnum())

def format_prediction_response(
    prediction: float,
    confidence: float,
    model_type: str,
    model_version: str,
    symbol: str = None,
    date: str = None
) -> Dict[str, Any]:
    """
    Format prediction response.
    
    Args:
        prediction: Predicted price
        confidence: Confidence score
        model_type: Type of model used
        model_version: Version of the model
        symbol: Stock symbol (optional)
        date: Prediction date (optional)
        
    Returns:
        Formatted response dictionary
    """
    # Handle invalid float values
    def safe_float(value: float) -> float:
        """Convert float to JSON-safe value."""
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(np.clip(value, -1e10, 1e10))  # Clip to reasonable range
    
    return {
        "symbol": symbol,
        "date": date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_price": safe_float(prediction),
        "confidence": safe_float(confidence),
        "model_type": model_type,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

def create_sequence_data(
    data: np.ndarray,
    sequence_length: int
) -> tuple:
    """
    Create sequences for time series data.
    
    Args:
        data: Input data array
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse)
    }

def get_date_range(days: int = 7) -> tuple:
    """
    Get start and end dates for a given number of days.
    
    Args:
        days: Number of days
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date 