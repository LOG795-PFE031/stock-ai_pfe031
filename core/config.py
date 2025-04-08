"""
Configuration settings for the Stock AI system.
"""
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel

class DataConfig(BaseModel):
    """Data collection and storage configuration."""
    STOCK_DATA_DIR: Path = Path("data/stock")
    NEWS_DATA_DIR: Path = Path("data/news")
    LOGS_DIR: Path = Path("logs")
    STOCK_HISTORY_DAYS: int = 365
    NEWS_HISTORY_DAYS: int = 7
    MAX_NEWS_ARTICLES: int = 100
    UPDATE_INTERVAL: int = 60  # minutes

class ModelConfig(BaseModel):
    """Model configuration."""
    PREDICTION_MODELS_DIR: Path = Path("models/specific")
    PROPHET_MODELS_DIR: Path = Path("models/prophet")
    NEWS_MODELS_DIR: Path = Path("models/news")
    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    FEATURES: list = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'MACD_Signal'
    ]
    SEQUENCE_LENGTH: int = 60
    BATCH_SIZE: int = 1024
    EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.2

class APIConfig(BaseModel):
    """API configuration."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list = ["*"]
    API_VERSION: str = "1.0"

class RabbitMQConfig(BaseModel):
    """RabbitMQ configuration."""
    HOST: str = "localhost"
    PORT: int = 5672
    USER: str = "guest"
    PASSWORD: str = "guest"
    VHOST: str = "/"
    QUEUE_PREFIX: str = "stock_ai"

class Config:
    """Main configuration class."""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.rabbitmq = RabbitMQConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.STOCK_DATA_DIR,
            self.data.NEWS_DATA_DIR,
            self.data.LOGS_DIR,
            self.model.PREDICTION_MODELS_DIR,
            self.model.PROPHET_MODELS_DIR,
            self.model.NEWS_MODELS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Create and export global config instance
__all__ = ['config']
config = Config() 