"""
Data service for fetching and processing stock data.
"""

import asyncio
from datetime import datetime, timezone, time
from typing import Dict, Any, List, Optional
import pytz
from time import perf_counter

import pandas as pd
import pandas_market_calendars as mcal
import requests
from sqlalchemy import select, delete
import yfinance as yf


from core.utils import get_start_date_from_trading_days, get_latest_trading_day
from db.session import get_sync_session
from db.models.stock_price import StockPrice
from monitoring.prometheus_metrics import external_requests_total
from .base_service import BaseService


class DataService(BaseService):
    """Service for managing data collection and processing."""

    def __init__(self):
        super().__init__()
        self.logger = self.logger["data"]
        self.news_data_dir = self.config.data.NEWS_DATA_DIR

    async def initialize(self) -> None:
        """Initialize the data service."""
        try:
            # Create necessary directories
            self.news_data_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self.logger.info("Data service initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize data service: %s", str(e))
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Data service cleaned up successfully")
        except Exception as e:
            self.logger.error("Error during data service cleanup: %s", str(e))

    def get_stock_name(self, symbol: str) -> str:
        """
        Get the name of a stock given its symbol

        Args:
            symbol (str): Stock symbol

        Returns:
            str: The stock name
        """

        symbol = symbol.upper()
        data_file = self.config.data.DATA_ROOT_DIR / "stock_names.csv"

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

            self.logger.info("Collected stock name for %s", symbol)

            # Add new entry
            df.loc[symbol] = name

            # Save updated csv cache
            df.to_csv(data_file)

            return name

        except Exception as e:
            self.logger.error("Error collecting stock name for %s: %s", symbol, str(e))

            # Log the unsuccessful external request to Yahoo Finance (Prometheuss)
            external_requests_total.labels(site="yahoo_finance", result="error").inc()
            raise

    async def get_nasdaq_stocks(self) -> dict:
        """
        Fetch the NASDAQ 100 stocks sorted by price percentage change (descending)

        Returns:
            dict: A dictionary containing the count of symbols and the list of symbols
            from the NASDAQ 100 index.
        """

        def parse_percentage_change(pct_str):
            """Parse the percentage change of a stock (absolute value)"""
            try:
                per_change = abs(float(pct_str.strip("%").replace(",", "")))
                return per_change
            except:
                return 0.0

        self.logger.info("Starting NASDAQ 100 stocks data retrieval process")

        try:
            # Fetch the data
            url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            data = response.json()["data"]["data"]["rows"]

            # Sort the list by absolute percentageChange (descending)
            sorted_stocks = sorted(
                data,
                key=lambda x: parse_percentage_change(x.get("percentageChange", "0%")),
                reverse=True,
            )
            stocks_data = {"count": len(sorted_stocks), "data": sorted_stocks}

            self.logger.info("Retrieved NASDAQ 100 stocks data")

            # Return the stock data
            return stocks_data
        except Exception as e:
            self.logger.error("Error collecting NASDAQ 100 stocks data: %s", str(e))
            raise

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves the stock price for the given symbol on the latest trading day.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").

        Returns:
            pandas.DataFrame: A DataFrame containing the stock data for the latest trading day.
            str: Stock Company Name
        """
        try:
            self.validate_symbol(symbol)

            # Get start and end dates (as today)
            start_date = get_latest_trading_day()

            # Collect fresh stock data price
            df = await self._collect_stock_data(
                symbol, start_date=start_date, end_date=start_date, update_existing=True
            )

            # If no data available
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                return {
                    "symbol": symbol,
                    "stock_name": symbol,
                    "current_price": 0.0,
                    "date_str": None,
                    "message": "No data available for this symbol",
                }

            # Transform DataFrame to list of price objects
            price = self._transform_dataframe_to_prices(df)

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            # Create message
            message = "Current price retrieved successfully from fresh data"

            self.logger.info("Retrieved current stock price for %s", symbol)

            result = {
                "symbol": symbol,
                "stock_name": stock_name or symbol,
                "date_str": start_date.isoformat(),
                "current_price": price,
                "message": message,
            }

            return result
        except Exception as e:
            self.logger.error(
                "Error getting current stock price for %s: %s", symbol, str(e)
            )
            raise

    async def get_recent_data(self, symbol: str, days_back: int = None):
        """
        Retrieves recent stock data for a given symbol looking back a specified number of trading
        days.

        If no specific number of days is provided, the default lookback period from the
        configuration is used.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            days_back (int, optional): The number of trading days to look back from the current
                date. If not provided, the default lookback period from the configuration is used.

        Returns:
            tuple: A tuple containing:
                - `pd.DataFrame`: A DataFrame containing stock price data for the requested symbol
                    and date range.
                - `str`: The name of the stock corresponding to the given symbol.
        """
        try:
            # If there is no number of days back, use the default config lookback period days
            if days_back is None:
                days_back = self.config.data.LOOKBACK_PERIOD_DAYS

            # Get start and end dates
            end_date = datetime.now()
            start_date = get_start_date_from_trading_days(end_date, days_back)

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Ensure the Date column is properly converted to datetime
            if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = pd.to_datetime(df["Date"], utc=True)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info(
                "Retrieved recent stock prices for %s looking back for %d trading days",
                symbol,
                days_back,
            )

            return df, stock_name
        except Exception as e:
            self.logger.error(
                "Error getting recent stock prices for %s looking back for %d trading days : %s",
                symbol,
                days_back,
                str(e),
            )
            raise

    async def get_historical_stock_prices_from_end_date(
        self, symbol: str, end_date: str, days_back: int
    ):
        """
        Retrieve historical stock prices for a symbol from a specified end date, looking back a
        given number of trading days.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").
            end_date (str): The end date to retrieve stock data from.
            days_back: The number of trading days to look back from the end date.

        Returns:
            - (tuple): A tuple containing a DataFrame with stock data and the stock symbol name.
        """
        try:
            self.validate_symbol(symbol)
            self.validate_days_back(days_back)

            # Get correct end_date
            end_date_formatted = self.validate_date_format(end_date)

            # Replace the time of end_date to 9:30 AM
            modified_end_date = end_date_formatted.replace(
                hour=9, minute=30, second=0, microsecond=0
            )

            # Localize to US/Eastern
            eastern = pytz.timezone("US/Eastern")
            modified_end_date = eastern.localize(modified_end_date)

            # Check if the modified end_date is not in the future
            if modified_end_date <= datetime.now().astimezone(eastern):
                # If it's not in the future, update the original end_date
                end_date_formatted = modified_end_date
            else:
                # If it's in the future, leave end_date unchanged
                pass

            # Get the start date
            start_date = get_start_date_from_trading_days(end_date_formatted, days_back)

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date_formatted)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date_formatted.date()
            )
            df = df[mask]

            # Transform DataFrame to list of price objects
            prices = self._transform_dataframe_to_prices(df)

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info(
                "Retrieved recent stock prices for %s looking back for %d trading days from %s to %s",
                symbol,
                days_back,
                start_date.isoformat(),
                end_date_formatted.isoformat(),
            )

            return df, stock_name

        except Exception as e:
            self.logger.error(
                "Error getting historical stock data from end date %s for %s: %s",
                end_date,
                symbol,
                str(e),
            )
            raise

    async def get_historical_stock_prices(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical stock prices for a symbol. If data doesn't exist or is outdated, collect
        new data.

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

            # Ensure the Date column is properly converted to datetime
            if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = pd.to_datetime(df["Date"], utc=True)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Sort by date
            df = df.sort_values("Date")

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info("Retrieved historical stock data for %s", symbol)

            return df, stock_name

        except Exception as e:
            self.logger.error("Error getting stock data for %s: %s", symbol, str(e))
            raise

    async def _collect_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        update_existing=False,
    ) -> pd.DataFrame:
        """
        Collect stock data from Yahoo Finance and store it in the database

        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            update_existing (bool): Determines if we want to update the existing element of the
            table (default to False)

        Returns:
            DataFrame containing stock data
        """
        try:
            total_start = perf_counter()
            self.logger.debug(f"[START] Collecting stock data for {symbol}")

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = get_start_date_from_trading_days(end_date)

            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            end_date = datetime.combine(end_date, time.max)

            self.logger.debug(f"[DATES] start_date: {start_date}, end_date: {end_date}")

            # Download data from Yahoo Finance
            yahoo_start = perf_counter()
            self.logger.debug(f"[YF] Starting Yahoo Finance fetch for {symbol}")
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            yahoo_duration = perf_counter() - yahoo_start
            self.logger.debug(
                f"[YF] Finished Yahoo Finance fetch in {yahoo_duration:.2f}s"
            )

            # Prometheus logging
            external_requests_total.labels(site="yahoo_finance", result="success").inc()

            # Process DataFrame
            df = df.reset_index()
            self.logger.debug(f"[DF] Retrieved {len(df)} rows")

            # Get stock name
            stock_name = self.get_stock_name(symbol)
            self.logger.debug(f"[INFO] Resolved stock name: {stock_name}")

            # Interact with the database
            db_start = perf_counter()
            self.logger.debug("[DB] Starting DB session")
            # Use asyncio.to_thread to avoid blocking the event loop for DB calls
            await asyncio.to_thread(
                self._upsert_stock_data,
                symbol,
                start_date,
                end_date,
                df,
                stock_name,
                update_existing,
            )

            db_duration = perf_counter() - db_start
            total_duration = perf_counter() - total_start

            self.logger.info(
                f"[SUCCESS] Collected stock data for {symbol} "
                f"(Yahoo: {yahoo_duration:.2f}s, "
                f"DB: {db_duration:.2f}s, Total: {total_duration:.2f}s)"
            )

            return df

        except Exception as e:
            self.logger.error(f"[FAILURE] Collecting data for {symbol}: {str(e)}")
            external_requests_total.labels(site="yahoo_finance", result="error").inc()
            raise

    def _upsert_stock_data(
        self, symbol, start_date, end_date, df, stock_name, update_existing
    ):
        """
        Sync function to insert or update stock price data in the database.
        This function checks if stock price records already exist for the given symbol
        and date range. If not, it inserts new records. If `update_existing` is True,
        it updates existing records with the latest values from the DataFrame.
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            update_existing (bool): Determines if we want to update the existing element of the
            table (default to False)
        """
        SessionLocal = get_sync_session()
        with SessionLocal() as session:

            # Build the query
            query = select(StockPrice).where(
                StockPrice.stock_symbol == symbol.upper(),
                StockPrice.date >= start_date.date(),
                StockPrice.date <= end_date.date(),
            )
            result = session.execute(query)
            existing_rows = result.scalars().all()

            existing_map = {row.date: row for row in existing_rows}
            self.logger.debug(f"[DB] Found {len(existing_map)} existing records")

            new_entries = []

            for _, row in df.iterrows():
                date_only = row["Date"].date()
                if date_only not in existing_map:
                    new_entries.append(
                        StockPrice(
                            stock_symbol=symbol.upper(),
                            stock_name=stock_name,
                            date=date_only,
                            open=row.get("Open"),
                            high=row.get("High"),
                            low=row.get("Low"),
                            close=row.get("Close"),
                            volume=(
                                int(row.get("Volume", 0)) if row.get("Volume") else None
                            ),
                            dividends=row.get("Dividends"),
                            stock_splits=row.get("Stock Splits"),
                        )
                    )
                elif update_existing:
                    existing = existing_map[date_only]
                    existing.open = row.get("Open")
                    existing.high = row.get("High")
                    existing.low = row.get("Low")
                    existing.close = row.get("Close")
                    existing.volume = (
                        int(row["Volume"]) if row.get("Volume", 0) else None
                    )
                    existing.dividends = row.get("Dividends")
                    existing.stock_splits = row.get("Stock Splits")

            self.logger.debug(f"[DB] Adding {len(new_entries)} new entries")
            if new_entries:
                session.add_all(new_entries)

                self.logger.debug("[DB] Commit successful")

    def _transform_dataframe_to_prices(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Transform DataFrame to list of price dictionaries.

        Args:
            df: DataFrame containing stock price data

        Returns:
            List of price dictionaries
        """
        prices = []
        for _, row in df.iterrows():
            # Handle date formatting safely
            if hasattr(row.name, "strftime"):
                date_str = row.name.strftime("%Y-%m-%d")
            elif "Date" in row and hasattr(row["Date"], "strftime"):
                date_str = row["Date"].strftime("%Y-%m-%d")
            else:
                date_str = str(row.name)

            prices.append(
                {
                    "Date": date_str,
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": int(row["Volume"]),
                    "Dividends": float(row["Dividends"]),
                    "Stock_splits": float(row["Stock Splits"]),
                }
            )

        return prices

    def validate_symbol(self, symbol: str) -> None:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock symbol to validate

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not symbol.isalnum():
            raise ValueError(f"Invalid stock symbol: {symbol}")

    def validate_days_back(self, days_back: int) -> None:
        """
        Validate days_back parameter.

        Args:
            days_back: Number of days to validate

        Raises:
            ValueError: If days_back is invalid
        """
        if days_back <= 0 or days_back > 365:
            raise ValueError("days_back must be between 1 and 365")

    def validate_date_format(self, date_str: str) -> datetime:
        """
        Validate and parse date string.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date format is invalid
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    async def _get_stock_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ):
        """
        Retrieves stock data for a given symbol and date range.

        If the data exists and contains valid data in the db (postgres) for the requested date
        range, it is returned. Otherwise, new data is fetched via `collect_stock_data()`.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            start_date (datetime): The start date of the desired data range.
            end_date (datetime): The end date of the desired data range.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data for the specified symbol and date
                range.
        """
        try:
            SessionLocal = get_sync_session()
            with SessionLocal() as session:

                try:
                    # Query the prices between the start and end date for the symbol
                    query = select(StockPrice).where(
                        StockPrice.stock_symbol == symbol.upper(),
                        StockPrice.date >= start_date.date(),
                        StockPrice.date <= end_date.date(),
                    )
                    result = session.execute(query)
                    records = result.scalars().all()

                    if records:

                        # Convert to DataFrame
                        df = pd.DataFrame(
                            [
                                {
                                    "Date": r.date,
                                    "Open": (
                                        float(r.open) if r.open is not None else None
                                    ),
                                    "High": (
                                        float(r.high) if r.high is not None else None
                                    ),
                                    "Low": float(r.low) if r.low is not None else None,
                                    "Close": (
                                        float(r.close) if r.close is not None else None
                                    ),
                                    "Volume": r.volume,
                                    "Dividends": (
                                        float(r.dividends)
                                        if r.dividends is not None
                                        else None
                                    ),
                                    "Stock Splits": (
                                        float(r.stock_splits)
                                        if r.stock_splits is not None
                                        else None
                                    ),
                                }
                                for r in records
                            ]
                        )

                        # Convert dates to timezone-aware UTC
                        df["Date"] = pd.to_datetime(
                            df["Date"], format="mixed", utc=True
                        )

                        # Check if the cache needs to be reloaded
                        reload = self._is_cache_valid(df, start_date, end_date)

                        if not reload:
                            self.logger.info("Load data from cache for %s", symbol)
                            # Return data from cache
                            return df.sort_values("Date")
                except Exception as db_error:
                    self.logger.error(
                        "Error during DB query for symbol %s: %s", symbol, str(db_error)
                    )
                    raise

            # No data exists or need reload, collect new data
            df = await self._collect_stock_data(symbol, start_date, end_date)
            return df

        except Exception as e:
            self.logger.error("Error getting stock data for %s: %s", symbol, str(e))
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

        # Ensure the Date column is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], utc=True)

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
            deleted_count = 0

            # Create a new async SQLAlchemy session to interact with the database
            SessionLocal = get_sync_session()
            with SessionLocal() as session:

                try:
                    if symbol:

                        # Perform delete operation for the given symbol (asynchronously)
                        stmt = delete(StockPrice).where(
                            StockPrice.stock_symbol == symbol
                        )
                        result = session.execute(stmt)

                        # Get the count of deleted rows
                        deleted_count = result.rowcount
                        self.logger.info(
                            "Deleted %d records for symbol: %s", deleted_count, symbol
                        )

                    else:
                        # Perform delete operation for all given symbol (asynchronously)
                        stmt = delete(StockPrice)
                        result = session.execute(stmt)

                        # Get the count of deleted rows
                        deleted_count = result.rowcount
                        self.logger.info(
                            "Deleted all %d stock price records", deleted_count
                        )

                except Exception as db_error:
                    # Rollback in case of error
                    session.rollback()
                    self.logger.error("Error during DB cleanup: %s", str(db_error))
                    raise

                return {
                    "status": "success",
                    "symbol": symbol if symbol else "all",
                    "deleted_count": deleted_count,
                }

        except Exception as e:
            # Rollback in case of error
            await session.rollback()
            self.logger.error("Error during cleanup: %s", str(e))

            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        finally:
            # Close the session
            session.close()
