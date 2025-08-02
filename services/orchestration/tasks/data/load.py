from datetime import datetime
import httpx
from prefect import task
import pandas as pd
from core.config import config


@task(
    name="load_recent_stock_data",
    description="Load recent stock data for a given symbol using the provided data service.",
    retries=3,
    retry_delay_seconds=2,
)
async def load_recent_stock_data(symbol: str) -> pd.DataFrame:
    """
    Prefect task to load recent stock data for a given symbol.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data.
    """
    data_service_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stock/recent"
    params = {"symbol": symbol, "days_back": config.data.LOOKBACK_PERIOD_DAYS}

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(data_service_url, params=params)
        response.raise_for_status()
        json_response = response.json()

        data = pd.DataFrame(json_response["prices"])

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Response data is not a valid DataFrame")

        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    return pd.DataFrame(data)


@task(
    name="get_historical_stock_prices_from_end_date",
    description="Load historical stock data for a given symbol using an end date and a "
    + "specified number of days back, with the provided API endpoint.",
    retries=3,
    retry_delay_seconds=2,
)
async def load_historical_stock_prices_from_end_date(
    symbol: str, end_date: str, days_back: int
) -> pd.DataFrame:
    """
    Prefect task to load historical stock data for a given symbol, ending at a specified
    date and going back a given number of days.

    Args:
        symbol (str): Stock ticker symbol.
        end_date (str): The end date for the historical data.
        days_back (int): The number of days to look back from the end date.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """

    data_service_url = (
        f"http://{config.data.HOST}:{config.data.PORT}/data/stock/from-end-date"
    )
    params = {"symbol": symbol, "end_date": end_date, "days_back": days_back}
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(data_service_url, params=params)
        response.raise_for_status()
        json_response = response.json()

        data = pd.DataFrame(json_response["prices"])

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Response data is not a valid DataFrame")

    return pd.DataFrame(data)
