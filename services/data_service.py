"""
Data service for fetching and processing stock data.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone, time
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
import asyncio
import aiohttp

from services.base_service import BaseService
from core.config import config
from core.logging import logger
from core.utils import calculate_technical_indicators
from monitoring.prometheus_metrics import external_requests_total


class DataService(BaseService):
    """Service for managing data collection and processing."""

    def __init__(self):
        super().__init__()
        self.news_sources = {
            "reuters": "https://www.reuters.com/markets/companies/",
            "yahoo_finance": "https://finance.yahoo.com/quote/{}/news",
        }
        self.logger = logger["data"]
        self.config = config
        self.stock_data_dir = self.config.data.STOCK_DATA_DIR
        self.news_data_dir = self.config.data.NEWS_DATA_DIR

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
                start_date = end_date - timedelta(
                    days=self.config.data.STOCK_HISTORY_DAYS
                )

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

            # Reset index to make Date a column
            df = df.reset_index()

            # Ensure Date column is timezone-aware UTC
            df["Date"] = pd.to_datetime(df["Date"], format="mixed", utc=True)

            # Calculate technical indicators
            df = calculate_technical_indicators(df)

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

    async def collect_news_data(
        self, symbol: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Collect news articles from various sources.

        Args:
            symbol: Stock symbol
            days: Number of days of news to collect

        Returns:
            List of dictionaries containing news articles
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Collect news from different sources
            tasks = [
                self._collect_reuters_news(symbol, start_date, end_date),
                self._collect_yahoo_news(symbol, start_date, end_date),
            ]

            results = await asyncio.gather(*tasks)

            # Combine and deduplicate news
            all_news = []
            seen_urls = set()

            for source_news in results:
                for article in source_news:
                    if article["url"] not in seen_urls:
                        seen_urls.add(article["url"])
                        all_news.append(article)

            # Sort by date
            all_news.sort(key=lambda x: x["published_date"], reverse=True)

            # Save news data
            news_file = self.news_data_dir / f"{symbol}_news.json"
            pd.DataFrame(all_news).to_json(news_file, orient="records")

            self.logger.info(f"Collected {len(all_news)} news articles for {symbol}")
            return all_news

        except Exception as e:
            self.logger.error(f"Error collecting news data for {symbol}: {str(e)}")
            raise

    async def _collect_reuters_news(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Collect news from Reuters."""
        try:
            url = f"{self.news_sources['reuters']}{symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Log the successful external request to Reuters (Prometheus)
                        external_requests_total.labels(
                            site="reuters", result="success"
                        ).inc()

                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        articles = []
                        for article in soup.find_all("article")[
                            : self.config.data.MAX_NEWS_ARTICLES
                        ]:
                            try:
                                title = article.find("h3").text.strip()
                                link = article.find("a")["href"]
                                date_str = article.find("time")["datetime"]
                                date = datetime.fromisoformat(
                                    date_str.replace("Z", "+00:00")
                                )

                                if start_date <= date <= end_date:
                                    articles.append(
                                        {
                                            "title": title,
                                            "url": link,
                                            "published_date": date,
                                            "source": "reuters",
                                        }
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error parsing Reuters article: {str(e)}"
                                )
                                continue

                        return articles
                    else:
                        self.logger.warning(
                            f"Failed to fetch Reuters news for {symbol}"
                        )

                        # Log the unsuccessful external request to Reuters (Prometheus)
                        external_requests_total.labels(
                            site="reuters", result="error"
                        ).inc()
                        return []

        except Exception as e:
            self.logger.error(f"Error collecting Reuters news for {symbol}: {str(e)}")
            return []

    async def _collect_yahoo_news(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Collect news from Yahoo Finance."""
        try:
            url = self.news_sources["yahoo_finance"].format(symbol)
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Log the successful external request to Yahoo Finance (Prometheus)
                        external_requests_total.labels(
                            site="yahoo_finance", result="success"
                        ).inc()

                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        articles = []
                        for article in soup.find_all(
                            "div", {"class": "js-stream-content"}
                        )[: self.config.data.MAX_NEWS_ARTICLES]:
                            try:
                                title = article.find("h3").text.strip()
                                link = article.find("a")["href"]
                                date_str = article.find(
                                    "span", {"class": "C(#959595)"}
                                ).text
                                date = datetime.strptime(date_str, "%b %d, %Y %I:%M %p")

                                if start_date <= date <= end_date:
                                    articles.append(
                                        {
                                            "title": title,
                                            "url": link,
                                            "published_date": date,
                                            "source": "yahoo_finance",
                                        }
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error parsing Yahoo Finance article: {str(e)}"
                                )
                                continue

                        return articles
                    else:
                        self.logger.warning(
                            f"Failed to fetch Yahoo Finance news for {symbol}"
                        )
                        # Log the unsuccessful external request to Yahoo Finance (Prometheus)
                        external_requests_total.labels(
                            site="yahoo_finance", result="error"
                        ).inc()

                        return []

        except Exception as e:
            self.logger.error(
                f"Error collecting Yahoo Finance news for {symbol}: {str(e)}"
            )
            return []

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

            # Update news data
            news_articles = await self.collect_news_data(
                symbol, days=self.config.data.NEWS_HISTORY_DAYS
            )

            return {
                "symbol": symbol,
                "stock_data": {
                    "rows": len(stock_df),
                    "latest_date": stock_df["Date"].max().isoformat(),
                },
                "news_data": {
                    "articles": len(news_articles),
                    "latest_date": (
                        max(
                            article["published_date"] for article in news_articles
                        ).isoformat()
                        if news_articles
                        else None
                    ),
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
                df = calculate_technical_indicators(df)

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

            # Calculate technical indicators if not present
            if "RSI" not in df.columns:
                df["RSI"] = self._calculate_rsi(df["Close"])

            if "MACD" not in df.columns:
                macd, signal = self._calculate_macd(df["Close"])
                df["MACD"] = macd
                df["MACD_Signal"] = signal

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

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

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
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = end_date - timedelta(
                    days=self.config.data.STOCK_HISTORY_DAYS
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

                # Get all the dates (business days) within start_date and end_date
                expected_dates = set(
                    pd.date_range(start=start_date, end=end_date, freq="B").date
                )

                # Extract the dates present in the data
                df_dates = set(df["Date"].dt.date)

                # Check if all expected dates are present
                missing_data = not expected_dates.issubset(df_dates)

                if data_is_stale or missing_data:
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
