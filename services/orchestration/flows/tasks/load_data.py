from prefect import task
import pandas as pd

from services import DataService


@task
def load_data(service: DataService, symbol: str) -> pd.DataFrame:
    data, _ = service.get_stock_data(symbol)
    return data
