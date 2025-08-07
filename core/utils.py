"""
Utility functions for the Stock AI system.
"""

import pandas as pd
import pandas_market_calendars as mcal
import pytz
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone, time
import logging
from core.config import config

logger = logging.getLogger(__name__)


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
    symbol: str,
    date: str,
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
        "status": "success",
        "symbol": symbol,
        "date": date.strftime("%Y-%m-%d"),
        "predicted_price": safe_float(prediction),
        "confidence": safe_float(confidence),
        "model_type": model_type,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
    }


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
        end = datetime.combine(
            datetime.now(timezone.utc).date(), time.max, tzinfo=timezone.utc
        )
        days = days or config.data.LOOKBACK_PERIOD_DAYS
        start = end - timedelta(days=days)

    return start, end


def get_latest_trading_day():
    """
    Get the latest valid trading day

    Returns:
        str: the latest valid trading day (in string format)
    """
    nyse = mcal.get_calendar("NYSE")
    eastern = pytz.timezone("US/Eastern")

    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    now_est = now_utc.astimezone(eastern)
    today = now_est.date()

    # Look back over the past 10 days to find the most recent trading day
    start_date = today - timedelta(days=10)
    end_date = today

    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    # Find the latest trading day that is today or before
    past_trading_days = schedule.index.date
    latest_trading_day = max(d for d in past_trading_days if d <= today)

    # Return the latest valid trading day
    return datetime.combine(latest_trading_day, datetime.min.time())


def get_next_trading_day(date: datetime = None) -> datetime:
    """
    Get the next valid trading day with the provided date

    Args:
        date (datetime): Provided date to look for next trading day

    Returns:
        str: the next valid trading day (in string format)
    """
    nyse = mcal.get_calendar("NYSE")
    eastern = pytz.timezone("US/Eastern")

    if date is None:
        # If there is no provided date, we look for next trading day from today
        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        date = now_utc.astimezone(eastern)
    elif date.tzinfo is None:
        # Make naive datetime Eastern-aware
        date = eastern.localize(date)
    else:
        # Normalize to Eastern
        date = date.astimezone(eastern)

    # Get the next few trading days (starting the day after the provided date)
    schedule = nyse.schedule(
        start_date=date + timedelta(days=1), end_date=date + timedelta(days=10)
    )

    # Return the first valid trading day after the provided date
    return schedule.index[0].to_pydatetime()


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

    return start_date.to_pydatetime()


def get_model_name(model_type: str, symbol: str):
    """
    Generate a standardized model name by combining the model type
    and stock symbol.

    Args:
        model_type (str): The model type
        symbol (str): The stock ticker symbol

    Returns:
        str: A string representing the model name
    """
    return f"{model_type}_{symbol}"
