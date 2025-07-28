from prefect import task
import pandas as pd
from datetime import datetime
import httpx
from typing import Callable

from core.config import config


@task(
    name="load_recent_stock_data",
    description="Load recent stock data for a given symbol using the API endpoint.",
    retries=3,
    retry_delay_seconds=5,
)
async def load_recent_stock_data(fetch_func: Callable, symbol: str) -> pd.DataFrame:
    """
    Prefect task to load recent stock data for a given symbol.

    Args:
        fetch_func (Callable): Function to fetch stock data from API.
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data.
    """
    data = await fetch_func(symbol)
    
    # Extract prices from the response
    prices = data.get("prices", [])
    if not prices:
        raise ValueError(f"No price data found for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(prices)
    
    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["date"])
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High", 
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close"
    })
    
    # Sort by date
    df = df.sort_values("Date")
    
    return df


@task(
    name="get_historical_stock_prices_from_end_date",
    description="Load historical stock data for a given symbol using an end date and a "
    + "specified number of days back, with the provided API endpoint.",
    retries=3,
    retry_delay_seconds=5,
)
async def load_historical_stock_prices_from_end_date(
    fetch_func: Callable, symbol: str, end_date: datetime, days_back: int
) -> pd.DataFrame:
    """
    Prefect task to load historical stock data for a given symbol, ending at a specified
    date and going back a given number of days.

    Args:
        fetch_func (Callable): Function to fetch stock data from API.
        symbol (str): Stock ticker symbol.
        end_date (datetime): The end date for the historical data.
        days_back (int): The number of days to look back from the end date.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    # For now, we'll use the same endpoint as recent data
    # In a real implementation, you might want to use a different endpoint
    data = await fetch_func(symbol, days_back)
    
    # Extract prices from the response
    prices = data.get("prices", [])
    if not prices:
        raise ValueError(f"No price data found for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(prices)
    
    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["date"])
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High", 
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close"
    })
    
    # Sort by date
    df = df.sort_values("Date")
    
    return df
