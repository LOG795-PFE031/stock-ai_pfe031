"""
Core configuration module for the stock prediction system.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    SEQUENCE_LENGTH: int = 60
    BATCH_SIZE: int = 1024
    EPOCHS: int = 50
    FEATURES: list = None
    
    def __post_init__(self):
        self.FEATURES = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'MACD_Signal'
        ]

@dataclass
class DataConfig:
    """Configuration for data handling"""
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    def __post_init__(self):
        # Create necessary directories
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                        self.MODELS_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ messaging"""
    HOST: str = "rabbitmq"
    PORT: int = 5672
    USER: str = "guest"
    PASSWORD: str = "guest"
    EXCHANGE: str = "quote-exchange"
    QUEUE: str = "stock-quotes"

@dataclass
class APIConfig:
    """Configuration for API endpoints"""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

class Config:
    """Main configuration class that combines all configs"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.rabbitmq = RabbitMQConfig()
        self.api = APIConfig()
        
        # Environment variables override
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        # Model config
        if os.getenv('SEQUENCE_LENGTH'):
            self.model.SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH'))
        if os.getenv('BATCH_SIZE'):
            self.model.BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
        if os.getenv('EPOCHS'):
            self.model.EPOCHS = int(os.getenv('EPOCHS'))
            
        # Data config
        if os.getenv('RAW_DATA_DIR'):
            self.data.RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')
        if os.getenv('PROCESSED_DATA_DIR'):
            self.data.PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR')
        if os.getenv('MODELS_DIR'):
            self.data.MODELS_DIR = os.getenv('MODELS_DIR')
        if os.getenv('LOGS_DIR'):
            self.data.LOGS_DIR = os.getenv('LOGS_DIR')
            
        # RabbitMQ config
        if os.getenv('RABBITMQ_HOST'):
            self.rabbitmq.HOST = os.getenv('RABBITMQ_HOST')
        if os.getenv('RABBITMQ_PORT'):
            self.rabbitmq.PORT = int(os.getenv('RABBITMQ_PORT'))
        if os.getenv('RABBITMQ_USER'):
            self.rabbitmq.USER = os.getenv('RABBITMQ_USER')
        if os.getenv('RABBITMQ_PASS'):
            self.rabbitmq.PASSWORD = os.getenv('RABBITMQ_PASS')
            
        # API config
        if os.getenv('API_HOST'):
            self.api.HOST = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.PORT = int(os.getenv('API_PORT'))
        if os.getenv('API_DEBUG'):
            self.api.DEBUG = os.getenv('API_DEBUG').lower() == 'true'

# Create a global config instance
config = Config() 