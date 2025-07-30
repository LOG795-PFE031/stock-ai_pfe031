"""
Data service for fetching and processing stock data.
"""

import asyncio
from datetime import datetime, timezone, time
from typing import Dict, Any, Optional, List
import time as py_time

import pandas as pd
import pandas_market_calendars as mcal
import requests
from sqlalchemy import select, delete
import yfinance as yf


from core.utils import get_start_date_from_trading_days, get_latest_trading_day
# Import the new session and model for the dedicated stock database
from services.data_ingestion.db.session import get_stock_async_session
from services.data_ingestion.db.models.stock_price import StockPrice
from core.prometheus_metrics import external_requests_total
from core import BaseService
from core.logging import logger
from core.config import config


class DataService(BaseService):
    """Service for managing data collection and processing."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.logger = logger["data"]
        self.config = config
        self.news_data_dir = self.config.data.NEWS_DATA_DIR

    async def initialize(self) -> None:
        """Initialize the data service."""
        try:
            # Create necessary directories
            self.news_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Test connection to the stock database
            try:
                AsyncSessionLocal = get_stock_async_session()
                async with AsyncSessionLocal() as session:
                    # Simple query to test database connection and create tables if needed
                    try:
                        stmt = select(StockPrice).limit(1)
                        await session.execute(stmt)
                        self.logger.info("✅ Connection to stock database verified successfully")
                    except Exception as table_error:
                        # Table might not exist yet, try to create it
                        self.logger.warning(f"⚠️ Stock prices table not found, attempting to create: {table_error}")
                        try:
                            from services.data_ingestion.db.init_db import init_stock_db
                            await init_stock_db()
                            self.logger.info("✅ Database tables created successfully")
                        except Exception as create_error:
                            self.logger.error(f"❌ Failed to create database tables: {create_error}")
                            raise
            except Exception as db_error:
                self.logger.error(f"❌ Error connecting to stock database: {str(db_error)}")
                raise
            
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
        Retrieve a list of NASDAQ-100 stocks, sorted by absolute percentage change in
        descending order (top movers first).
        
        This method fetches data from the NASDAQ API, sorts stocks based on their absolute
        percentage price change (regardless of direction), and includes company names
        for each stock.

        Returns:
            dict: A dictionary containing:
                - count (int): Number of stocks in the NASDAQ-100 index
                - data (list): List of stock dictionaries, each containing symbol, name,
                  price, and percentage change, sorted with top movers first
        """

        def parse_percentage_change(pct_str):
            """
            Parse the percentage change string into a float and convert to absolute value.
            This is used for sorting stocks by their movement magnitude regardless of direction.
            
            Args:
                pct_str (str): Percentage change as a string (e.g. "5.2%", "-3.1%", "UNCH")
                
            Returns:
                float: Absolute value of the percentage change, or 0.0 for non-numeric values
            """
            try:
                # Handle non-numeric values like 'UNCH'
                if not pct_str or pct_str.strip() in ['UNCH', 'N/A', '--', '']:
                    return 0.0
                
                # Remove % sign and commas, convert to float, and get absolute value
                per_change = abs(float(pct_str.strip("%").replace(",", "")))
                return per_change
            except (ValueError, TypeError):
                # Return 0.0 for any conversion errors
                return 0.0

        self.logger.info("Starting NASDAQ-100 stocks data retrieval process")

        try:
            # Fetch the data from NASDAQ API
            url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
            headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/92.0.4515.107 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json, text/plain, */*"
            }
            
            # Log the API request attempt
            self.logger.debug(f"Making request to NASDAQ API: {url}")
            
            # Make the request and parse the JSON response
            response = requests.get(url, headers=headers,timeout=10)
            
            # Log the success of the external request
            external_requests_total.labels(site="nasdaq_api", result="success").inc()
            
            # Extract the rows containing stock data
            data = response.json()["data"]["data"]["rows"]

            # Sort the list by absolute percentageChange (descending)
            sorted_stocks = sorted(
                data,
                key=lambda x: parse_percentage_change(x.get("percentageChange", "0%")),
                reverse=True,
            )

            for stock in sorted_stocks:
                symbol = stock.get("symbol")
                try:
                    stock["name"] = self.get_stock_name(symbol)
                except Exception:
                    stock["name"] = None

            stocks_data = {"count": len(sorted_stocks), "data": sorted_stocks}

            self.logger.info("Retrieved NASDAQ 100 stocks data")

            # Return the stock data
            return stocks_data
        except requests.RequestException as re:
            # Handle specific request exceptions
            self.logger.error("Network error when retrieving NASDAQ-100 stocks data: %s", str(re))
            # Log the failed external request
            external_requests_total.labels(site="nasdaq_api", result="error").inc()
            raise
        except (KeyError, ValueError) as parse_error:
            # Handle JSON parsing errors
            self.logger.error("Error parsing NASDAQ API response: %s", str(parse_error))
            external_requests_total.labels(site="nasdaq_api", result="error").inc()
            raise
        except Exception as e:
            # Handle other unexpected errors
            self.logger.error("Unexpected error collecting NASDAQ-100 stocks data: %s", str(e))
            external_requests_total.labels(site="nasdaq_api", result="error").inc()
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
            # Get a broader date range to ensure we get recent data
            end_date = datetime.now(timezone.utc)
            start_date = get_start_date_from_trading_days(end_date, 10)  # Look back 10 trading days

            self.logger.debug(f"Getting current price for {symbol} from {start_date.date()} to {end_date.date()}")

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning(f"No data found for {symbol} between {start_date.date()} and {end_date.date()}")
                return df, self.get_stock_name(symbol)

            # Ensure Date column is datetime type
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # If Date is the index, convert it to a column
                df = df.reset_index()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    self.logger.error(f"Date column not found in DataFrame for {symbol}")
                    return df, self.get_stock_name(symbol)

            # Get the most recent data point
            df = df.sort_values('Date', ascending=False)
            current_price = df.iloc[0:1]  # Get the most recent row
            most_recent_date = current_price.iloc[0]['Date']
            
            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info("Retrieved current stock price for %s at the date %s", symbol, most_recent_date)


            return current_price, stock_name
        except Exception as e:
            self.logger.error(
                "Error getting current stock price for %s: %s", symbol, str(e)
            )
            raise
        
    async def calculate_change_percent(self, symbol: str) -> float:
        """
        Calculate the percentage change between the current price and previous day's close.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Percentage change rounded to 2 decimal places, or None if calculation fails
        """
        try:
            # Get recent data with more days to ensure we have enough data points
            recent_data, _ = await self.get_recent_data(symbol, days_back=10)
            
            if recent_data.empty:
                self.logger.warning(f"No data available to calculate change percent for {symbol}")
                return None
                
            # Log the shape of the data for debugging
            self.logger.debug(f"Retrieved {len(recent_data)} rows for {symbol} to calculate change percent")
            
            # Ensure the Date column is a datetime
            if 'Date' in recent_data.columns:
                recent_data['Date'] = pd.to_datetime(recent_data['Date'])
                
                # Sort by date (most recent first)
                recent_data = recent_data.sort_values('Date', ascending=False)
            else:
                # If Date is the index, convert it to a column
                recent_data = recent_data.reset_index()
                if 'Date' in recent_data.columns:
                    recent_data['Date'] = pd.to_datetime(recent_data['Date'])
                    # Sort by date (most recent first)
                    recent_data = recent_data.sort_values('Date', ascending=False)
                else:
                    self.logger.error(f"Date column not found in DataFrame for {symbol}")
                    return None
            
            # Check if we have at least 2 data points
            if len(recent_data) < 2:
                self.logger.warning(f"Not enough data points ({len(recent_data)}) to calculate change percent for {symbol}")
                return None
            
            # Get current day and previous day prices
            current_close = float(recent_data.iloc[0]['Close'])
            previous_close = float(recent_data.iloc[1]['Close'])
            
            # Log the values being used for calculation
            self.logger.debug(f"{symbol} change calculation: current={current_close}, previous={previous_close}")
            
            # Calculate percentage change
            if previous_close > 0:
                change_percent = ((current_close - previous_close) / previous_close) * 100
                result = round(change_percent, 2)
                self.logger.info(f"Calculated change percent for {symbol}: {result}%")
                return result
            else:
                self.logger.warning(f"Previous close price for {symbol} is zero or negative: {previous_close}")
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating change percent for {symbol}: {str(e)}")
            self.logger.debug(f"Exception details: {repr(e)}")
            return None

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
            
            # Ensure days_back is reasonable
            if days_back < 1:
                days_back = 1
                self.logger.debug(f"Adjusted days_back to {days_back} for minimum data points")
            
            # Get the exact number of trading days requested
            end_date = datetime.now(timezone.utc)
            start_date = get_start_date_from_trading_days(end_date, days_back)
            
            self.logger.debug(f"Requesting exactly {days_back} trading days for {symbol} from {start_date.date()} to {end_date.date()}")

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"No data retrieved for {symbol} between {start_date.date()} and {end_date.date()}")
                return df, self.get_stock_name(symbol)

            # Ensure Date column is datetime type
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # If Date is the index, convert it to a column
                df = df.reset_index()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    self.logger.error(f"Date column not found in DataFrame for {symbol}")
                    return pd.DataFrame(), self.get_stock_name(symbol)

            # Sort by date (most recent first) and limit to exactly the requested number of days
            df = df.sort_values('Date', ascending=False)
            df = df.head(days_back)  # Get only the exact number of days requested
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column {col} in data for {symbol}")
                    return pd.DataFrame(), self.get_stock_name(symbol)
            
            # Sort back to chronological order (oldest first)
            df = df.sort_values('Date', ascending=True)
            
            # Log the number of rows retrieved
            self.logger.debug(f"Retrieved exactly {len(df)} rows of data for {symbol} (requested {days_back})")

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info(
                "Retrieved recent stock prices for %s looking back for %d trading days (got %d data points)",
                symbol,
                days_back,
                len(df)
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
        self, symbol: str, end_date: datetime, days_back: int
    ):
        """
        Retrieve historical stock prices for a symbol from a specified end date, looking back a
        given number of trading days.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").
            end_date: The end date to retrieve stock data from.
            days_back: The number of trading days to look back from the end date.

        Returns:
            - (tuple): A tuple containing a DataFrame with stock data and the stock symbol name.
        """
        try:
            # Ensure end date is timezone-aware
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Get the start date
            start_date = get_start_date_from_trading_days(end_date, days_back)

            # Retrieve stock data prices
            df = await self._get_stock_data(symbol, start_date, end_date)

            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning(f"No data found for {symbol} between {start_date.date()} and {end_date.date()}")
                return df, self.get_stock_name(symbol)

            # Ensure Date column is datetime type
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # If Date is the index, convert it to a column
                df = df.reset_index()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    self.logger.error(f"Date column not found in DataFrame for {symbol}")
                    return df, self.get_stock_name(symbol)

            # Filter data for requested date range
            mask = (df["Date"].dt.date >= start_date.date()) & (
                df["Date"].dt.date <= end_date.date()
            )
            df = df[mask]

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            self.logger.info(
                "Retrieved recent stock prices for %s looking back for %d trading days from %s to %s",
                symbol,
                days_back,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
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

            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning(f"No data found for {symbol} between {start_date.date()} and {end_date.date()}")
                return df

            # Ensure Date column is datetime type
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # If Date is the index, convert it to a column
                df = df.reset_index()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    self.logger.error(f"Date column not found in DataFrame for {symbol}")
                    return df

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
        start_time = py_time.time()
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

            # Log the date range being requested
            self.logger.debug(f"Requesting Yahoo Finance data for {symbol} from {start_date.date()} to {end_date.date()}")

            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)

            # Retrieve the data using asyncio.to_thread
            df = await asyncio.to_thread(stock.history, start=start_date, end=end_date)
            # Log timing for Yahoo Finance request
            yahoo_start = py_time.time()
            df = await asyncio.to_thread(stock.history, start=start_date, end=end_date)
            yahoo_time = py_time.time() - yahoo_start
            self.logger.info(f"Yahoo Finance request for {symbol} took {yahoo_time:.2f} seconds")

            # Check if we got any data
            if df.empty:
                self.logger.warning(f"No data returned from Yahoo Finance for {symbol} between {start_date.date()} and {end_date.date()}")
                # Log the unsuccessful external request to Yahoo Finance (Prometheus)
                external_requests_total.labels(site="yahoo_finance", result="error").inc()
                return df

            # Log the successful external request to Yahoo Finance (Prometheus)
            external_requests_total.labels(site="yahoo_finance", result="success").inc()

            # Reset index to make Date a column
            df = df.reset_index()

            # Log the data we received
            self.logger.debug(f"Received {len(df)} rows of data for {symbol}")

            # Get the stock_name
            stock_name = self.get_stock_name(symbol)

            # Create a new async SQLAlchemy session to interact with the database
            AsyncSessionLocal = get_stock_async_session()
            async with AsyncSessionLocal() as session:

                try:
                    # Get all the prices given the symbol
                    query = select(StockPrice).where(
                        StockPrice.stock_symbol == symbol.upper()
                    )
                    result = await session.execute(query)

                    # Get all the existing dates in the db
                    existing_dates = {row.date for row in result.scalars().all()}

                    new_entries = []

                    for _, row in df.iterrows():
                        # Check if there is already an entry in the db (given the symbol ticker
                        # and the date)
                        date_only = row["Date"].date()

                        if date_only not in existing_dates:
                            # Add the new entry
                            new_entries.append(
                                StockPrice(
                                    stock_symbol=symbol.upper(),
                                    stock_name=stock_name,
                                    date=row["Date"].date(),
                                    open=row.get("Open"),
                                    high=row.get("High"),
                                    low=row.get("Low"),
                                    close=row.get("Close"),
                                    volume=(
                                        int(row.get("Volume", 0))
                                        if row.get("Volume")
                                        else None
                                    ),
                                    dividends=row.get("Dividends"),
                                    stock_splits=row.get("Stock Splits"),
                                )
                            )

                    # Add all the new entries to the db
                    if new_entries:
                        session.add_all(new_entries)
                        # Commit the transaction
                        await session.commit()
                        self.logger.info(f"Added {len(new_entries)} new entries to database for {symbol}")
                    else:
                        self.logger.debug(f"No new entries to add for {symbol}")

                except Exception as db_error:
                    # Rollback in case of error
                    await session.rollback()

                    self.logger.error(
                        "Error while interacting with the database: %s", str(db_error)
                    )
                    raise
            self.logger.info("Collected stock data for %s", symbol)
            total_time = py_time.time() - start_time
            self.logger.info(f"Total data collection for {symbol} took {total_time:.2f} seconds")
            return df

        except Exception as e:
            self.logger.error("Error collecting stock data for %s: %s", symbol, str(e))

            # Log the unsuccessful external request to Yahoo Finance (Prometheus)
            external_requests_total.labels(site="yahoo_finance", result="error").inc()
            raise

    async def verify_yahoo_finance_data(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Verify if data is available in Yahoo Finance for a given symbol.
        
        Args:
            symbol: Stock symbol to verify
            days_back: Number of days to look back
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Set date range
            end_date = datetime.now(timezone.utc)
            start_date = get_start_date_from_trading_days(end_date, days_back)
            
            self.logger.info(f"Verifying Yahoo Finance data availability for {symbol}")
            self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)
            
            # Get basic info first
            info = await asyncio.to_thread(stock.info)
            
            # Get historical data
            df = await asyncio.to_thread(stock.history, start=start_date, end=end_date)
            
            result = {
                "symbol": symbol,
                "date_range": {
                    "start": start_date.date().isoformat(),
                    "end": end_date.date().isoformat()
                },
                "data_available": not df.empty,
                "rows_returned": len(df),
                "columns": list(df.columns) if not df.empty else [],
                "info_available": bool(info),
                "company_name": info.get("longName", "Unknown") if info else "Unknown",
                "sector": info.get("sector", "Unknown") if info else "Unknown",
                "industry": info.get("industry", "Unknown") if info else "Unknown"
            }
            
            if not df.empty:
                result["date_range_data"] = {
                    "first_date": df.index[0].date().isoformat() if len(df) > 0 else None,
                    "last_date": df.index[-1].date().isoformat() if len(df) > 0 else None,
                    "sample_data": {
                        "Open": float(df.iloc[-1]["Open"]) if len(df) > 0 else None,
                        "High": float(df.iloc[-1]["High"]) if len(df) > 0 else None,
                        "Low": float(df.iloc[-1]["Low"]) if len(df) > 0 else None,
                        "Close": float(df.iloc[-1]["Close"]) if len(df) > 0 else None,
                        "Volume": int(df.iloc[-1]["Volume"]) if len(df) > 0 else None
                    }
                }
            
            self.logger.info(f"Verification result for {symbol}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error verifying Yahoo Finance data for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "data_available": False
            }

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
            AsyncSessionLocal = get_stock_async_session()
            async with AsyncSessionLocal() as session:

                try:
                    # Query the prices between the start and end date for the symbol
                    query = select(StockPrice).where(
                        StockPrice.stock_symbol == symbol.upper(),
                        StockPrice.date >= start_date,
                        StockPrice.date <= end_date,
                    )
                    result = await session.execute(query)
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
        # Ensure Date column is datetime type
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # If Date is the index, convert it to a column
            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                self.logger.error("Date column not found in DataFrame for cache validation")
                return set()  # Return empty set to indicate cache is invalid

        # Get the trading days within start_date and end_date
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_dates = set(schedule["market_open"].dt.date)

        # Extract the dates present in the data
        df_dates = set(df["Date"].dt.date)

        # Check if all expected dates are present
        missing_dates = trading_dates - df_dates

        return missing_dates

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the data service.

        Returns:
            Dictionary containing health status
        """
        try:
            # Check if service is initialized
            if not self._initialized:
                return {
                    "status": "unhealthy",
                    "message": "Service not initialized",
                    "timestamp": datetime.now().isoformat(),
                }

            # Check database connectivity
            try:
                AsyncSessionLocal = get_stock_async_session()
                async with AsyncSessionLocal() as session:
                    # Simple query to test database connection
                    stmt = select(StockPrice).limit(1)
                    await session.execute(stmt)
                    
                return {
                    "status": "healthy",
                    "message": "Service is healthy",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as db_error:
                return {
                    "status": "unhealthy",
                    "message": f"Database connection failed: {str(db_error)}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

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

            # Create a new async SQLAlchemy session to interact with the dedicated stock database
            AsyncSessionLocal = get_stock_async_session()
            async with AsyncSessionLocal() as session:

                try:
                    if symbol:

                        # Perform delete operation for the given symbol (asynchronously)
                        stmt = delete(StockPrice).where(
                            StockPrice.stock_symbol == symbol
                        )
                        result = await session.execute(stmt)

                        # Get the count of deleted rows
                        deleted_count = result.rowcount
                        self.logger.info(
                            "Deleted %d records for symbol: %s", deleted_count, symbol
                        )

                    else:
                        # Perform delete operation for all given symbol (asynchronously)
                        stmt = delete(StockPrice)
                        result = await session.execute(stmt)

                        # Get the count of deleted rows
                        deleted_count = result.rowcount
                        self.logger.info(
                            "Deleted all %d stock price records", deleted_count
                        )

                    # Commit the transaction if records were deleted
                    await session.commit()

                except Exception as db_error:
                    # Rollback in case of error
                    await session.rollback()
                    self.logger.error("Error during DB cleanup: %s", str(db_error))
                    raise

                return {
                    "message": f"Successfully deleted {deleted_count} records",
                    "deleted_records": deleted_count,
                    "symbols_affected": [symbol] if symbol else ["all"],
                }

        except Exception as e:
            self.logger.error("Error during cleanup: %s", str(e))

            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def pre_populate_popular_stocks(self, symbols: List[str] = None, days_back: int = 365) -> Dict[str, Any]:
        """
        Pre-populate the database with popular stocks to improve access speed.
        
        Args:
            symbols: List of stock symbols to pre-populate. If None, uses default popular stocks.
            days_back: Number of days of historical data to fetch.
            
        Returns:
            Dictionary containing pre-population results
        """
        if symbols is None:
            # Default popular stocks from NASDAQ-100
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "CSCO", "QCOM",
                "AVGO", "TXN", "MU", "AMAT", "ADP", "COST", "SBUX", "MDLZ",
                "GILD", "REGN", "VRTX", "ABNB", "ZM", "SNPS", "KLAC", "LRCX"
            ]
        
        results = {
            "total_symbols": len(symbols),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        self.logger.info(f"Starting pre-population of {len(symbols)} popular stocks")
        
        for symbol in symbols:
            try:
                # Get historical data which will trigger database population
                end_date = datetime.now(timezone.utc)
                start_date = get_start_date_from_trading_days(end_date, days_back)
                
                await self._get_stock_data(symbol, start_date, end_date)
                results["successful"] += 1
                self.logger.info(f"✅ Pre-populated data for {symbol}")
                
            except Exception as e:
                results["failed"] += 1
                error_msg = f"Failed to pre-populate {symbol}: {str(e)}"
                results["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(f"Pre-population completed: {results['successful']} successful, {results['failed']} failed")
        return results

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

    def _validate_price_data(self, data_row) -> bool:
        """
        Validate that price data is accurate and reasonable.
        
        Args:
            data_row: DataFrame row containing price data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check if all required fields are present and numeric
            required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
            for field in required_fields:
                if field not in data_row or pd.isna(data_row[field]):
                    return False
                if not isinstance(data_row[field], (int, float)) or data_row[field] <= 0:
                    return False
            
            # Validate price relationships
            open_price = float(data_row['Open'])
            high_price = float(data_row['High'])
            low_price = float(data_row['Low'])
            close_price = float(data_row['Close'])
            volume = int(data_row['Volume'])
            
            # Basic sanity checks
            if (high_price < low_price or 
                high_price < open_price or 
                high_price < close_price or
                low_price > open_price or 
                low_price > close_price):
                return False
            
            # Check for reasonable price ranges (stocks shouldn't be $0 or $1M+)
            if close_price < 0.01 or close_price > 1000000:
                return False
            
            # Check for reasonable volume (should be positive)
            if volume <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating price data: {str(e)}")
            return False

    def _validate_dataframe_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate the overall quality of a DataFrame.
        
        Args:
            df: DataFrame containing stock data
            
        Returns:
            bool: True if data quality is good, False otherwise
        """
        try:
            if df.empty:
                return False
            
            # Check required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    return False
            
            # Check for non-null data
            if df[required_columns].isnull().any().any():
                return False
            
            # Check for reasonable number of rows (at least 2 for change calculation)
            if len(df) < 2:
                return False
            
            # Check for reasonable date range (not too old or future dates)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                min_date = df['Date'].min()
                max_date = df['Date'].max()
                current_date = pd.Timestamp.now()
                
                # Data shouldn't be more than 10 years old or in the future
                if min_date < (current_date - pd.Timedelta(days=3650)) or max_date > current_date:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrame quality: {str(e)}")
            return False

    def _calculate_change_percent_from_dataframe(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate percentage change from existing DataFrame data.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'Close' column
            
        Returns:
            Optional[float]: Percentage change rounded to 2 decimal places, or None if calculation fails
        """
        try:
            if df.empty or len(df) < 2:
                self.logger.warning("Not enough data points to calculate change percent")
                return None
            
            # Ensure Date column is datetime and sort by date (most recent first)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df_sorted = df.sort_values('Date', ascending=False)
            else:
                # If Date is the index, convert it to a column
                df = df.reset_index()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df_sorted = df.sort_values('Date', ascending=False)
                else:
                    self.logger.error("Date column not found in DataFrame")
                    return None
            
            # Get current day and previous day prices
            current_close = float(df_sorted.iloc[0]['Close'])
            previous_close = float(df_sorted.iloc[1]['Close'])
            
            # Log the values being used for calculation
            self.logger.debug(f"Change calculation: current={current_close}, previous={previous_close}")
            
            # Calculate percentage change
            if previous_close > 0:
                change_percent = ((current_close - previous_close) / previous_close) * 100
                result = round(change_percent, 2)
                self.logger.info(f"Calculated change percent: {result}%")
                return result
            else:
                self.logger.warning(f"Previous close price is zero or negative: {previous_close}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating change percent from DataFrame: {str(e)}")
            return None

    async def get_current_price_with_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price with all metadata and optimized change percent calculation.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing current price data and metadata
        """
        try:
            self.validate_symbol(symbol)
            
            # Force fresh data collection from Yahoo Finance for accuracy
            end_date = datetime.now(timezone.utc)
            start_date = get_start_date_from_trading_days(end_date, 5)  # Get last 5 trading days
            
            # Get fresh data directly from Yahoo Finance
            df = await self._collect_stock_data(symbol, start_date, end_date)
            
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                result = {
                    "symbol": symbol,
                    "stock_name": symbol,
                    "current_price": 0.0,
                    "change_percent": None,
                    "date_str": None,
                    "message": "No data available for this symbol"
                }
            else:
                # Get the most recent data point
                df_sorted = df.sort_values('Date', ascending=False)
                latest_data = df_sorted.iloc[0]
                
                current_price = float(latest_data['Close'])
                stock_name = self.get_stock_name(symbol)
                
                # Calculate change percent from the same DataFrame (no additional API calls!)
                change_percent = self._calculate_change_percent_from_dataframe(df_sorted)
                
                # Prepare date string
                date_str = latest_data['Date'].strftime('%Y-%m-%d') if hasattr(latest_data['Date'], 'strftime') else str(latest_data['Date'])[:10]
                
                # Create message
                message = "Current price retrieved successfully from fresh data"
                if change_percent is None:
                    message = "Current price retrieved, but change percent calculation failed"
                
                result = {
                    "symbol": symbol,
                    "stock_name": stock_name or symbol,
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "date_str": date_str,
                    "message": message
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting current price with metadata for {symbol}: {str(e)}")
            raise

    async def get_historical_data_with_metadata(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical data with optimized change percent calculation.
        
        Args:
            symbol: Stock symbol
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Dictionary containing historical data and metadata
        """
        try:
            self.validate_symbol(symbol)
            
            # Parse dates
            start = self.validate_date_format(start_date)
            end = self.validate_date_format(end_date)
            
            # Get historical data
            data, stock_name = await self.get_historical_stock_prices(symbol, start, end)
            
            # Calculate change percent from the same data (no additional API calls!)
            change_percent = self._calculate_change_percent_from_dataframe(data)
            
            # Get current price from the most recent data point
            current_price = None
            if not data.empty:
                data_sorted = data.sort_values('Date', ascending=False)
                latest_data = data_sorted.iloc[0]
                if self._validate_price_data(latest_data):
                    current_price = float(latest_data['Close'])
            
            # Transform DataFrame to list of price objects
            prices = self._transform_dataframe_to_prices(data)
            
            return {
                "symbol": symbol,
                "stock_name": stock_name,
                "current_price": current_price,
                "change_percent": change_percent,
                "prices": prices,
                "total_records": len(prices),
                "start_date": start.isoformat(),
                "end_date": end.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data with metadata for {symbol}: {str(e)}")
            raise

    async def get_recent_data_with_metadata(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """
        Get recent data with optimized change percent calculation.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days back
            
        Returns:
            Dictionary containing recent data and metadata
        """
        try:
            self.validate_symbol(symbol)
            self.validate_days_back(days_back)
            
            # Force fresh data collection to ensure accuracy
            end_date = datetime.now(timezone.utc)
            start_date = get_start_date_from_trading_days(end_date, days_back)
            
            # Get fresh data directly from Yahoo Finance
            df = await self._collect_stock_data(symbol, start_date, end_date)
            
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                result = {
                    "symbol": symbol,
                    "stock_name": symbol,
                    "current_price": 0.0,
                    "change_percent": None,
                    "prices": [],
                    "total_records": 0,
                    "days_back": days_back
                }
            else:
                # Get stock name
                stock_name = self.get_stock_name(symbol)
                
                # Get current price from the most recent data
                df_sorted = df.sort_values('Date', ascending=False)
                if not df_sorted.empty:
                    latest_data = df_sorted.iloc[0]
                    current_price = float(latest_data['Close'])
                    # Calculate change percent from the same DataFrame (no additional API calls!)
                    change_percent = self._calculate_change_percent_from_dataframe(df_sorted)
                else:
                    current_price = 0.0
                    change_percent = None
                
                # Transform DataFrame to list of price objects
                prices = self._transform_dataframe_to_prices(df)
                
                result = {
                    "symbol": symbol,
                    "stock_name": stock_name,
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "prices": prices,
                    "total_records": len(prices),
                    "days_back": days_back
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting recent data with metadata for {symbol}: {str(e)}")
            raise

    async def get_data_from_end_date_with_metadata(self, symbol: str, end_date: str, days_back: int) -> Dict[str, Any]:
        """
        Get data from end date with optimized change percent calculation.
        
        Args:
            symbol: Stock symbol
            end_date: End date string (YYYY-MM-DD)
            days_back: Number of days back from end date
            
        Returns:
            Dictionary containing data and metadata
        """
        try:
            self.validate_symbol(symbol)
            self.validate_days_back(days_back)
            
            # Parse end date
            end = self.validate_date_format(end_date)
            
            # Get data from end date - UNPACK the tuple!
            data, stock_name = await self.get_historical_stock_prices_from_end_date(symbol, end, days_back)
            
            # Calculate change percent from the same data (no additional API calls!)
            change_percent = self._calculate_change_percent_from_dataframe(data)
            
            # Get current price from the most recent data point
            current_price = None
            if not data.empty:
                data_sorted = data.sort_values('Date', ascending=False)
                latest_data = data_sorted.iloc[0]
                current_price = float(latest_data['Close'])
            
            # Transform DataFrame to list of price objects
            prices = self._transform_dataframe_to_prices(data)
            
            return {
                "symbol": symbol,
                "stock_name": stock_name,
                "current_price": current_price,
                "change_percent": change_percent,
                "prices": prices,
                "total_records": len(prices),
                "end_date": end.isoformat(),
                "days_back": days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data from end date with metadata for {symbol}: {str(e)}")
            raise

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
            if hasattr(row.name, 'strftime'):
                date_str = row.name.strftime('%Y-%m-%d')
            elif 'Date' in row and hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row.name)
                
            prices.append({
                "date": date_str,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']),
                "adj_close": float(row['Adj Close']) if 'Adj Close' in row else None
            })
        
        return prices
