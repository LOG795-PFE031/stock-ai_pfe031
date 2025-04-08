"""
Test configuration and fixtures.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from keras import Sequential, layers  # Replace tensorflow import
from prophet import Prophet
from typing import Dict, Any

from core.config import Config
from services.data_service import DataService
from services.model_service import ModelService
from services.training_service import TrainingService

@pytest.fixture
def config():
    """Create a test configuration."""
    test_config = Config()
    
    # Override paths for testing
    test_config.data.STOCK_DATA_DIR = Path("tests/data/stock")
    test_config.data.NEWS_DATA_DIR = Path("tests/data/news")
    test_config.model.PREDICTION_MODELS_DIR = Path("tests/models/prediction")
    test_config.model.NEWS_MODELS_DIR = Path("tests/models/news")
    
    # Create test directories
    test_config.data.STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_config.data.NEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_config.model.PREDICTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    test_config.model.NEWS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    yield test_config
    
    # Cleanup test directories
    shutil.rmtree(test_config.data.STOCK_DATA_DIR.parent)
    shutil.rmtree(test_config.model.PREDICTION_MODELS_DIR.parent)

@pytest.fixture
def sample_stock_data():
    """Create sample stock data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(150, 250, len(dates)),
        'Low': np.random.uniform(50, 150, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_news_data():
    """Create sample news data."""
    return [
        {
            "title": "Test Article 1",
            "url": "https://example.com/1",
            "published_date": datetime.now().isoformat(),
            "source": "reuters"
        },
        {
            "title": "Test Article 2",
            "url": "https://example.com/2",
            "published_date": (datetime.now() - timedelta(days=1)).isoformat(),
            "source": "yahoo_finance"
        }
    ]

@pytest.fixture
def sample_lstm_model():
    """Create a sample LSTM model."""
    model = Sequential([
        layers.LSTM(50, input_shape=(60, 13)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@pytest.fixture
def sample_prophet_model():
    """Create a sample Prophet model."""
    return Prophet()

@pytest.fixture
async def data_service(config):
    """Create a test data service."""
    service = DataService()
    await service.initialize()
    yield service
    await service.cleanup()

@pytest.fixture
async def model_service(config):
    """Create a test model service."""
    service = ModelService()
    await service.initialize()
    yield service
    await service.cleanup()

@pytest.fixture
async def training_service(config, data_service, model_service):
    """Create a test training service."""
    service = TrainingService(model_service=model_service, data_service=data_service)
    await service.initialize()
    yield service
    await service.cleanup()

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 