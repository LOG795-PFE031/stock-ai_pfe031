from prefect import task
import pandas as pd
from datetime import datetime

from services import DataService


@task(
    name="load_recent_stock_data",
    description="Load recent stock data for a given symbol using the provided data service.",
    retries=3,
    retry_delay_seconds=5,
)
async def load_recent_stock_data(service: DataService, symbol: str) -> pd.DataFrame:
    """
    Prefect task to load recent stock data for a given symbol.

    Args:
        service (DataService): Data service.
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data.
    """
    data, _ = await service.get_recent_data(symbol)
    return data


@task(
    name="get_historical_stock_prices_from_end_date",
    description="Load historical stock data for a given symbol using an end date and a "
    + "specified number of days back, with the provided data service.",
    retries=3,
    retry_delay_seconds=5,
)
async def load_historical_stock_prices_from_end_date(
    service: DataService, symbol: str, end_date: datetime, days_back: int
) -> pd.DataFrame:
    """
    Prefect task to load historical stock data for a given symbol, ending at a specified
    date and going back a given number of days.

    Args:
        service (DataService): Data service.
        symbol (str): Stock ticker symbol.
        end_date (datetime): The end date for the historical data.
        days_back (int): The number of days to look back from the end date.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    data, _ = await service.get_historical_stock_prices_from_end_date(
        symbol, end_date=end_date, days_back=days_back
    )
    return data
