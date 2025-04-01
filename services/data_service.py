"""
Data service for fetching and processing stock data.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from pathlib import Path

from services.base_service import BaseService
from core.config import config
from core.logging import logger
from core.utils import calculate_technical_indicators

class DataService(BaseService):
    """Service for managing data collection and processing."""
    
    def __init__(self):
        super().__init__()
        self.news_sources = {
            "reuters": "https://www.reuters.com/markets/companies/",
            "yahoo_finance": "https://finance.yahoo.com/quote/{}/news"
        }
        self.logger = logger['data']
    
    async def initialize(self) -> None:
        """Initialize the data service."""
        try:
            # Create necessary directories
            self.config.data.STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.data.NEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    
    async def collect_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
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
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.config.data.STOCK_HISTORY_DAYS)
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Save data
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            df.to_csv(data_file, index=False)
            
            self.logger.info(f"Collected stock data for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting stock data for {symbol}: {str(e)}")
            raise
    
    async def collect_news_data(
        self,
        symbol: str,
        days: int = 7
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
                self._collect_yahoo_news(symbol, start_date, end_date)
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
            news_file = self.config.data.NEWS_DATA_DIR / f"{symbol}_news.json"
            pd.DataFrame(all_news).to_json(news_file, orient="records")
            
            self.logger.info(f"Collected {len(all_news)} news articles for {symbol}")
            return all_news
            
        except Exception as e:
            self.logger.error(f"Error collecting news data for {symbol}: {str(e)}")
            raise
    
    async def _collect_reuters_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Collect news from Reuters."""
        try:
            url = f"{self.news_sources['reuters']}{symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        articles = []
                        for article in soup.find_all('article')[:self.config.data.MAX_NEWS_ARTICLES]:
                            try:
                                title = article.find('h3').text.strip()
                                link = article.find('a')['href']
                                date_str = article.find('time')['datetime']
                                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                
                                if start_date <= date <= end_date:
                                    articles.append({
                                        "title": title,
                                        "url": link,
                                        "published_date": date,
                                        "source": "reuters"
                                    })
                            except Exception as e:
                                self.logger.warning(f"Error parsing Reuters article: {str(e)}")
                                continue
                        
                        return articles
                    else:
                        self.logger.warning(f"Failed to fetch Reuters news for {symbol}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error collecting Reuters news for {symbol}: {str(e)}")
            return []
    
    async def _collect_yahoo_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Collect news from Yahoo Finance."""
        try:
            url = self.news_sources['yahoo_finance'].format(symbol)
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        articles = []
                        for article in soup.find_all('div', {'class': 'js-stream-content'})[:self.config.data.MAX_NEWS_ARTICLES]:
                            try:
                                title = article.find('h3').text.strip()
                                link = article.find('a')['href']
                                date_str = article.find('span', {'class': 'C(#959595)'}).text
                                date = datetime.strptime(date_str, '%b %d, %Y %I:%M %p')
                                
                                if start_date <= date <= end_date:
                                    articles.append({
                                        "title": title,
                                        "url": link,
                                        "published_date": date,
                                        "source": "yahoo_finance"
                                    })
                            except Exception as e:
                                self.logger.warning(f"Error parsing Yahoo Finance article: {str(e)}")
                                continue
                        
                        return articles
                    else:
                        self.logger.warning(f"Failed to fetch Yahoo Finance news for {symbol}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error collecting Yahoo Finance news for {symbol}: {str(e)}")
            return []
    
    async def update_data(
        self,
        symbol: str,
        update_interval: Optional[int] = None
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
                symbol,
                days=self.config.data.NEWS_HISTORY_DAYS
            )
            
            return {
                "symbol": symbol,
                "stock_data": {
                    "rows": len(stock_df),
                    "latest_date": stock_df['Date'].max().isoformat()
                },
                "news_data": {
                    "articles": len(news_articles),
                    "latest_date": max(
                        article["published_date"] for article in news_articles
                    ).isoformat() if news_articles else None
                },
                "timestamp": datetime.now().isoformat()
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
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            if not data_file.exists():
                # If file doesn't exist, collect new data
                df = await self.collect_stock_data(symbol)
            else:
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
            # Get the last 60 days of data (enough for technical indicators)
            df = df.tail(60)
            
            return {
                "status": "success",
                "data": df
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_historical_data(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to get
            
        Returns:
            Dictionary containing historical data
        """
        try:
            # Load data from file
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            if not data_file.exists():
                # If file doesn't exist, collect new data
                df = await self.collect_stock_data(symbol)
            else:
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
            # Get the last n days of data
            end_date = df['Date'].max()
            start_date = end_date - pd.Timedelta(days=days)
            historical_data = df[df['Date'] >= start_date]
            
            return {
                "status": "success",
                "data": historical_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            } 