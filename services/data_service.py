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
from core.utils import get_start_date_from_trading_days
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

    async def update_data(
        self, symbol: str, update_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update both stock and news data for a symbol.

        Args:
            symbol: Stock symbol
            update_interval: Update interval in minutes

        Returns:
            Dictionary containing update results
        """
        try:
            # Update stock data
            stock_df = await self.collect_stock_data(symbol)

            return {
                "symbol": symbol,
                "stock_data": {
                    "rows": len(stock_df),
                    "latest_date": stock_df["Date"].max().isoformat(),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {str(e)}")
            raise

    async def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest stock data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary containing the latest data
        """
        try:
            # Load data from file
            data_file = self.stock_data_dir / f"{symbol}_data.csv"

            # Check if file exists and is valid
            needs_refresh = False
            if not data_file.exists():
                needs_refresh = True
            else:
                try:
                    df = pd.read_csv(data_file)

                    # Convert date column to timezone-aware UTC
                    df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

                    # Validate data
                    if len(df) < 80:  # Need at least 80 days for technical indicators
                        self.logger.warning(
                            f"Data file for {symbol} has insufficient data points: {len(df)}"
                        )
                        needs_refresh = True
                    elif (
                        datetime.now(timezone.utc) - df["Date"].max()
                    ).days > 1:  # Data is more than 1 day old
                        self.logger.warning(
                            f"Data file for {symbol} is outdated: {df['Date'].max()}"
                        )
                        needs_refresh = True
                    elif not all(
                        col in df.columns for col in self.config.model.FEATURES
                    ):
                        self.logger.warning(
                            f"Data file for {symbol} is missing required columns"
                        )
                        needs_refresh = True
                except Exception as e:
                    self.logger.error(f"Error reading data file for {symbol}: {str(e)}")
                    needs_refresh = True

            # Refresh data if needed
            if needs_refresh:
                self.logger.info(f"Refreshing data for {symbol}")
                df = await self.collect_stock_data(symbol)
            else:
                # Recalculate technical indicators
                # TODO We need to separate the preprocessing step
                # df = calculate_technical_indicators(df)
                pass

            # Get enough data for technical indicators (60 days + max lookback period)
            # MA_20 needs 20 days, so we need at least 80 days to get 60 complete sequences
            df = df.tail(80)

            # Final validation
            if len(df) < 80:
                raise ValueError(
                    f"Failed to collect sufficient data for {symbol}. Got {len(df)} days, need 80."
                )

            return {"status": "success", "data": df}

        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_historical_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical stock data for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of days of historical data to include

        Returns:
            Dictionary containing the data and metadata
        """
        try:
            # Read data file
            file_path = self.stock_data_dir / f"{symbol}_data.csv"
            if not file_path.exists():
                self.logger.error(f"Data file not found for {symbol}")
                return {"status": "error", "error": f"Data file not found for {symbol}"}

            df = pd.read_csv(file_path)

            # Convert date column to datetime with proper timezone handling
            if "Date" in df.columns:
                # Use format='mixed' to handle different date formats
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

            # Sort by date and get last n days
            df = df.sort_values("Date", ascending=False)
            df = df.head(days)

            return {
                "status": "success",
                "data": df,
                "metadata": {
                    "symbol": symbol,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "data_points": len(df),
                    "date_range": {
                        "start": df["Date"].min().isoformat(),
                        "end": df["Date"].max().isoformat(),
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"Error reading data file for {symbol}: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get stock data for a symbol. If data doesn't exist or is outdated, collect new data.

        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            DataFrame containing stock data
        """
        try:
            # Get the eastern timezone
            eastern_timezone = pytz.timezone("America/New_York")

            # Set end_date if not provided
            end_date = end_date or datetime.now(eastern_timezone)

            if not start_date:
                business_days = pd.bdate_range(
                    end=end_date, periods=self.config.data.LOOKBACK_PERIOD_DAYS
                )
                start_date = (
                    business_days[0].to_pydatetime().astimezone(eastern_timezone)
                )

            # Ensure dates are timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Check if we have recent data
            data_file = self.stock_data_dir / f"{symbol}_data.csv"

            # Get the stock_name
            stock_name = await self.get_stock_name(symbol)

            if data_file.exists():
                df = pd.read_csv(data_file)
                # Convert dates to timezone-aware UTC
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

                # Check if data is up to date
                latest_date = df["Date"].max()
                current_date = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                data_is_stale = (
                    latest_date.date() < (current_date - timedelta(days=1)).date()
                )

                # Get the trading days within start_date and end_date
                nyse = mcal.get_calendar("NYSE")
                schedule = nyse.schedule(start_date=start_date, end_date=end_date)
                trading_dates = set(schedule["market_open"].dt.date)

                # Extract the dates present in the data
                df_dates = set(df["Date"].dt.date)

                # Check if all expected dates are present
                missing_dates = trading_dates - df_dates

                if data_is_stale or missing_dates:
                    # Update data
                    df = await self.collect_stock_data(symbol, start_date, end_date)
            else:
                # No data exists, collect new data
                df = await self.collect_stock_data(symbol, start_date, end_date)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Sort by date
            df = df.sort_values("Date")

            self.logger.info(f"Retrieved stock data for {symbol}")
            return df, stock_name

        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            raise

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
                    elif (datetime.now() - df["Date"].max()).days > 1:
                        self.logger.warning(
                            f"Data file {data_file.name} is outdated: {df['Date'].max()}"
                        )
                        needs_cleanup = True
                    elif not all(
                        col in df.columns for col in self.config.model.FEATURES
                    ):
                        self.logger.warning(
                            f"Data file {data_file.name} is missing required columns"
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
