"""
Utility functions for the Stock AI system.
"""

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from core.config import config

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
        df["Returns"] = df["Close"].pct_change()

        # Calculate Moving Averages
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()

        # Calculate Volatility (20-day standard deviation of returns)
        df["Volatility"] = df["Returns"].rolling(window=20).std()

        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Ensure Adj Close exists (if not, use Close)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Replace NaN values with forward fill then backward fill
        df = df.ffill().bfill()

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
    date: str = None,
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
        "timestamp": datetime.now().isoformat(),
    }


def create_sequence_data(data: np.ndarray, sequence_length: int) -> tuple:
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
        X.append(data[i : (i + sequence_length)])
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

    return {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}


def get_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
) -> tuple:
    """
    Get start and end dates for a given number of days or specific date range.

    Args:
        start_date: Optional start date string in ISO format
        end_date: Optional end date string in ISO format
        days: Optional number of days (used if start_date and end_date are not provided)

    Returns:
        Tuple of (start_date, end_date)
    """
    if start_date and end_date:
        # Parse provided dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    else:
        # Use days parameter or default to config
        end = datetime.now()
        days = days or config.data.LOOKBACK_PERIOD_DAYS
        start = end - timedelta(days=days)

    return start, end


def get_next_trading_day():
    """
    Get the next valid trading day

    Returns:
        str: the next valid trading day (in string format)
    """
    nyse = mcal.get_calendar("NYSE")
    today = datetime.now()

    # Get the next few trading days (starting tomorrow)
    schedule = nyse.schedule(
        start_date=today + timedelta(days=1), end_date=today + timedelta(days=10)
    )

    # Return the first valid trading day after today
    next_day = schedule.index[0]
    return next_day.strftime("%Y-%m-%d")


def get_start_date_from_trading_days(
    end_date: datetime, lookback_days: int = config.data.LOOKBACK_PERIOD_DAYS
) -> datetime:
    """
    Calculate the start date that is a specified number of NYSE trading days before a given end date.

    Args:
        end_date (datetime): The end date of the trading period (inclusive).
        lookback_days (int, optional): The number of trading days to look back from
            the end date. Defaults to config.data.STOCK_HISTORY_DAYS.

    Returns:
        datetime: The start date that is `lookback_days` NYSE trading sessions before `end_date`.
    """
    nyse = mcal.get_calendar("NYSE")

    # Estimate a large enough date range to capture the required number of trading days
    estimated_range_days = lookback_days * 2
    rough_start = end_date - timedelta(days=estimated_range_days)

    # Get valid NYSE trading days between the rough start and end date
    sessions = nyse.valid_days(rough_start, end_date)

    # Ensure we have at least the desired number of trading days
    if len(sessions) < lookback_days:
        raise ValueError(
            f"Only {len(sessions)} trading days available, need {lookback_days}."
        )

    # Select the start date that is exactly `lookback_days` trading sessions before the end date
    start_date = sessions[-lookback_days]

    return start_date
