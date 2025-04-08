"""
Tests for the data service.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

pytestmark = pytest.mark.asyncio

async def test_data_service_initialization(data_service, config):
    """Test data service initialization."""
    assert data_service._initialized
    assert Path(config.data.STOCK_DATA_DIR).exists()
    assert Path(config.data.NEWS_DATA_DIR).exists()

async def test_collect_stock_data(data_service, config):
    """Test stock data collection."""
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = await data_service.collect_stock_data(symbol, start_date, end_date)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Date" in df.columns
    assert "Close" in df.columns
    assert "Volume" in df.columns
    
    # Check if technical indicators are calculated
    assert "SMA_20" in df.columns
    assert "RSI" in df.columns
    assert "MACD" in df.columns
    
    # Check if data is saved
    data_file = config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
    assert data_file.exists()

async def test_collect_news_data(data_service, config):
    """Test news data collection."""
    symbol = "AAPL"
    days = 7
    
    articles = await data_service.collect_news_data(symbol, days)
    
    assert isinstance(articles, list)
    assert len(articles) > 0
    
    for article in articles:
        assert "title" in article
        assert "url" in article
        assert "published_date" in article
        assert "source" in article
    
    # Check if data is saved
    news_file = config.data.NEWS_DATA_DIR / f"{symbol}_news.json"
    assert news_file.exists()

async def test_update_data(data_service):
    """Test data update functionality."""
    symbol = "AAPL"
    
    result = await data_service.update_data(symbol)
    
    assert isinstance(result, dict)
    assert "symbol" in result
    assert "stock_data" in result
    assert "news_data" in result
    assert "timestamp" in result
    
    assert result["symbol"] == symbol
    assert isinstance(result["stock_data"], dict)
    assert isinstance(result["news_data"], dict)
    assert isinstance(result["timestamp"], str)

async def test_collect_stock_data_invalid_symbol(data_service):
    """Test stock data collection with invalid symbol."""
    symbol = "INVALID_SYMBOL"
    
    with pytest.raises(Exception):
        await data_service.collect_stock_data(symbol)

async def test_collect_news_data_invalid_symbol(data_service):
    """Test news data collection with invalid symbol."""
    symbol = "INVALID_SYMBOL"
    
    articles = await data_service.collect_news_data(symbol)
    assert len(articles) == 0

async def test_collect_stock_data_date_range(data_service):
    """Test stock data collection with specific date range."""
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = await data_service.collect_stock_data(symbol, start_date, end_date)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) <= 8  # Account for non-trading days

async def test_collect_news_data_limit(data_service, config):
    """Test news data collection with article limit."""
    symbol = "AAPL"
    days = 1
    
    articles = await data_service.collect_news_data(symbol, days)
    
    assert len(articles) <= config.data.MAX_NEWS_ARTICLES

async def test_data_service_cleanup(data_service):
    """Test data service cleanup."""
    await data_service.cleanup()
    assert not data_service._initialized 