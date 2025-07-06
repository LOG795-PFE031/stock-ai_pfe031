"""
Data service for fetching and processing stock data.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone, time
import pytz
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import requests
import json

from .base_service import BaseService
from core.config import config
from core.logging import logger
from core.utils import get_start_date_from_trading_days, get_latest_trading_day
from monitoring.prometheus_metrics import external_requests_total


class DataService(BaseService):
    """Service for managing data collection and processing."""

    def __init__(self):
        super().__init__()
        self.logger = logger["data"]
        self.config = config
        self.stock_data_dir = self.config.data.STOCK_DATA_DIR
        self.news_data_dir = self.config.data.NEWS_DATA_DIR
        self.symbols_file = self.config.data.NASDAQ100_SYMBOLS_FILE

    async def initialize(self) -> None:
        """Initialize the data service."""
        try:
            # Create necessary directories
            self.stock_data_dir.mkdir(parents=True, exist_ok=True)
            self.news_data_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self.logger.info("Data service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data service: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Data service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during data service cleanup: {str(e)}")

    async def get_stock_name(self, symbol: str) -> str:
        """
        Get the name of a stock given its symbol

        Args:
            symbol (str): Stock symbol

        Returns:
            str: The stock name
        """

        symbol = symbol.upper()
        data_file = self.stock_data_dir / "stock_names.csv"

        try:
            # Check if cache exist
            if data_file.exists():
                df = pd.read_csv(data_file, index_col="symbol")
                if symbol in df.index:
                    return df.loc[symbol, "name"]
            else:
                # Empty DataFrame with correct structure
                df = pd.DataFrame(columns=["name"])
                df.index.name = "symbol"

            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)

            # Log the successful external request to Yahoo Finance (Prometheus)
            external_requests_total.labels(site="yahoo_finance", result="success").inc()

            # Get the stock name (company)
            name = stock.info.get("shortName")

            self.logger.info(f"Collected stock name for {symbol}")

            # Add new entry
            df.loc[symbol] = name

            # Save updated csv cache
            df.to_csv(data_file)

            return name

        except Exception as e:
            self.logger.error(f"Error collecting stock name for {symbol}: {str(e)}")

            # Log the unsuccessful external request to Yahoo Finance (Prometheuss)
            external_requests_total.labels(site="yahoo_finance", result="error").inc()
            raise

    async def get_nasdaq_symbols(self) -> dict:
        """
        Fetch the NASDAQ 100 symbols

        Returns:
            dict: A dictionary containing the count of symbols and the list of symbols
            from the NASDAQ 100 index.
        """

        self.logger.info("Starting NASDAQ 100 symbol retrieval process")

        try:

            symbol_data_file = self.symbols_file

            if symbol_data_file.exists():

                self.logger.info("Loading NASDAQ 100 symbols from local cache")

                # Read the file
                with open(symbol_data_file) as f:
                    symbols_data = json.load(f)
            else:
                self.logger.info(
                    "Local cache not found, fetching NASDAQ 100 symbols from API"
                )

                # Fetch the data
                url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers)
                data = response.json()["data"]["data"]["rows"]

                # Retrieve the symbols
                symbols = [item["symbol"] for item in data]
                symbols_data = {"count": len(symbols), "symbols": symbols}

                # Save it to the data file
                with open(symbol_data_file, "w") as f:
                    json.dump(symbols_data, f, ensure_ascii=False)

            # Return the symbols
            self.logger.info("Retrieved NASDAQ 100 symbols")
            return symbols_data
        except Exception as e:
            self.logger.error(f"Error collecting NASDAQ 100 symbols: {str(e)}")
            raise

    async def collect_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Collect stock data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            DataFrame containing stock data
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = get_start_date_from_trading_days(end_date)

            # Ensure dates are timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Adjust the end date to the last possible moment of the day (to have the full-day)
            end_date = datetime.combine(end_date, time.max)

            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)

            # Log the successful external request to Yahoo Finance (Prometheus)
            external_requests_total.labels(site="yahoo_finance", result="success").inc()

            df = stock.history(start=start_date, end=end_date)

            # TODO Save raw data ?
            # data_file = self.stock_data_dir / f"raw_{symbol}.csv"
            # df.to_csv(data_file, index=False)

            # Reset index to make Date a column
            df = df.reset_index()

            # Save data
            data_file = self.stock_data_dir / f"{symbol}_data.csv"
            df.to_csv(data_file, index=False)

            self.logger.info(f"Collected stock data for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error collecting stock data for {symbol}: {str(e)}")

            # Log the unsuccessful external request to Yahoo Finance (Prometheus)
            external_requests_total.labels(site="yahoo_finance", result="error").inc()
            raise

    async def get_current_price(self, symbol: str):
        """
        Retrieves the stock price for the given symbol on the latest trading day.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").

        Returns:
            pandas.DataFrame: A DataFrame containing the stock data for the latest trading day.
            str: Stock Company Name
        """
        try:
            # Get start and end dates (as today)
            start_date = get_latest_trading_day()
            end_date = start_date

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Retrieve the latest trading day
            current_price = df[df["Date"].dt.date == end_date.date()]

            # Get the stock_name
            stock_name = await self.get_stock_name(symbol)

            self.logger.info(f"Retrieved current stock price for {symbol}")

            return current_price, stock_name
        except Exception as e:
            self.logger.error(
                f"Error getting current stock price for {symbol}: {str(e)}"
            )
            raise

    async def get_recent_data(self, symbol: str, days_back: int = None):
        try:
            # If there is no number of days back, use the default config lookback period days
            if days_back is None:
                days_back = self.config.data.LOOKBACK_PERIOD_DAYS

            # Get start and end dates
            end_date = datetime.now()
            start_date = get_start_date_from_trading_days(end_date, days_back)

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Get the stock_name
            stock_name = await self.get_stock_name(symbol)

            self.logger.info(
                f"Retrieved recent stock prices for {symbol} looking back for {days_back} trading days"
            )

            return df, stock_name
        except Exception as e:
            self.logger.error(
                f"Error getting recent stock prices for {symbol} looking back for {days_back} trading days : {str(e)}"
            )
            raise

    async def get_historical_stock_prices(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical stock prices for a symbol. If data doesn't exist or is outdated, collect new data.

        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            pd.DataFrame: DataFrame containing stock data
            str: Stock Company Name
        """
        try:
            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Ensure dates are timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Sort by date
            df = df.sort_values("Date")

            # Get the stock_name
            stock_name = await self.get_stock_name(symbol)

            self.logger.info(f"Retrieved historical stock data for {symbol}")

            return df, stock_name

        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            raise

    async def _get_stock_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ):
        """
        Retrieves stock data for a given symbol and date range, using a cached CSV file if available and valid.

        If a cached file exists and contains valid data for the requested date range, it is returned.
        Otherwise, new data is fetched via `collect_stock_data()`.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            start_date (datetime): The start date of the desired data range.
            end_date (datetime): The end date of the desired data range.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data for the specified symbol and date range.
        """
        try:
            # Check if we have recent data
            data_file = self.stock_data_dir / f"{symbol}_data.csv"

            if data_file.exists():
                df = pd.read_csv(data_file)

                # Convert dates to timezone-aware UTC
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

                # Check if the cache needs to be reloaded
                reload = self._is_cache_valid(df, start_date, end_date)

                if not reload:
                    self.logger.info(f"Load data from cache for {symbol}")
                    # Return data from cache
                    return df

            # No data exists or need reload, collect new data
            return await self.collect_stock_data(symbol, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            raise

    def _is_cache_valid(self, df, start_date, end_date):
        """
        Checks whether the cached stock data is valid by ensuring that all expected
        NYSE trading days within the given date range are present in the DataFrame.

        Args:
            df (pd.DataFrame): The cached stock data.
            start_date (datetime): The start of the date range to validate.
            end_date (datetime): The end of the date range to validate.

        Returns:
            set: A set of missing trading dates. If empty, the cache is considered valid.
        """
        # Get the trading days within start_date and end_date
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_dates = set(schedule["market_open"].dt.date)

        # Extract the dates present in the data
        df_dates = set(df["Date"].dt.date)

        # Check if all expected dates are present
        missing_dates = trading_dates - df_dates

        return missing_dates

    async def cleanup_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean up and maintain data files.

        Args:
            symbol: Optional specific symbol to clean up. If None, cleans all data files.

        Returns:
            Dictionary containing cleanup results
        """
        try:
            cleaned_files = []
            failed_files = []

            # Get list of files to clean
            if symbol:
                files = [self.stock_data_dir / f"{symbol}_data.csv"]
            else:
                files = list(self.stock_data_dir.glob("*_data.csv"))

            for data_file in files:
                try:
                    # Skip if file doesn't exist
                    if not data_file.exists():
                        continue

                    # Read and validate data
                    df = pd.read_csv(data_file)
                    df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

                    # Check for issues
                    needs_cleanup = False
                    if len(df) < 80:
                        self.logger.warning(
                            f"Data file {data_file.name} has insufficient data points: {len(df)}"
                        )
                        needs_cleanup = True
                    elif (
                        get_latest_trading_day().astimezone(timezone.utc)
                        - df["Date"].max()
                    ).days > 1:
                        self.logger.warning(
                            f"Data file {data_file.name} is outdated: {df['Date'].max()}"
                        )
                        needs_cleanup = True
                    elif df.isna().any().any():
                        self.logger.warning(
                            f"Data file {data_file.name} contains NaN values"
                        )
                        needs_cleanup = True

                    # Clean up if needed
                    if needs_cleanup:
                        # Backup the file
                        backup_file = data_file.with_suffix(".csv.bak")
                        data_file.rename(backup_file)

                        # Collect fresh data
                        symbol = data_file.stem.split("_")[0]
                        await self.collect_stock_data(symbol)

                        cleaned_files.append(data_file.name)

                except Exception as e:
                    self.logger.error(f"Error cleaning up {data_file.name}: {str(e)}")
                    failed_files.append(data_file.name)

            return {
                "status": "success",
                "cleaned_files": cleaned_files,
                "failed_files": failed_files,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error during data cleanup: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
