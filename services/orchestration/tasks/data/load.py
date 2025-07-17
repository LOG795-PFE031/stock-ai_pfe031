from prefect import task
import pandas as pd

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
