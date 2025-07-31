"""
Configuration settings for the Stock AI system.
"""

import os
from pathlib import Path
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Data collection and storage configuration."""

    DATA_ROOT_DIR: Path = Path("data")
    NEWS_DATA_DIR: Path = Path("data/news")
    LOOKBACK_PERIOD_DAYS: int = 365
    NEWS_HISTORY_DAYS: int = 7
    MAX_NEWS_ARTICLES: int = 100
    UPDATE_INTERVAL: int = 60  # minutes
    HOST: str = "data-service"
    PORT: int = 8000


class PreprocessingConfig(BaseModel):
    """Preprocessing service configuration"""

    SCALERS_DIR: Path = Path("data/scalers")
    SCALER_REGISTRY_JSON: Path = SCALERS_DIR / "scaler_registry.json"
    TRAINING_SPLIT_RATIO: float = 0.8
    SEQUENCE_LENGTH: int = 60


class ModelConfig(BaseModel):
    """Model configuration."""

    MODELS_ROOT_DIR: Path = Path("data/models")
    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    FEATURES: list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Returns",
        "MA_5",
        "MA_20",
        "Volatility",
        "RSI",
        "MACD",
        "MACD_Signal",
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

    HOST: str = "rabbitmq"  # Use the Docker container's hostname
    PORT: int = 5672
    USER: str = "guest"
    PASSWORD: str = "guest"
    VHOST: str = "/"
    QUEUE_PREFIX: str = "stock_ai"


class MLFlowConfig(BaseModel):
    """MLflow server configuration."""

    HOST: str = "mlflow-server"  # Use the Docker container's hostname
    PORT: int = 5000


class DataProcessingServiceConfig(BaseModel):
    """Data processing service configuration."""

    HOST: str = "data-processing-service"
    PORT: int = 8000


class TrainingServiceConfig(BaseModel):
    """Training service configuration."""

    HOST: str = "training-service"  # Use the Docker container's hostname
    PORT: int = 8000


class EvaluationServiceConfig(BaseModel):
    """Evaluation service configuration."""

    HOST: str = "evaluation-service"  # Use the Docker container's hostname
    PORT: int = 8000
    

class PostgresDatabaseConfig(BaseModel):
    """PostgreSQL configuration for the main database"""

    HOST: str = "postgres-stock-ai"
    PORT: int = 5432
    USER: str = "admin"
    PASSWORD: str = "admin"

    @property
    def URL(self) -> str:
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/stocks"


class StocksDatabaseConfig(BaseModel):
    """PostgreSQL configuration for the dedicated stock data database"""

    HOST: str = os.getenv("STOCK_DB_HOST", "postgres-stock-data")
    PORT: int = int(os.getenv("STOCK_DB_PORT", "5432"))
    USER: str = os.getenv("STOCK_DB_USER", "stockuser")
    PASSWORD: str = os.getenv("STOCK_DB_PASSWORD", "stockpass")
    DB_NAME: str = os.getenv("STOCK_DB_NAME", "stockdata")

    @property
    def URL(self) -> str:
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DB_NAME}"


class NewsServiceConfig(BaseModel):
    """News service configuration."""

    HOST: str = "news-service"
    PORT: int = 8000


class MonitoringConfig(BaseModel):
    """Monitoring service configuration."""

    # Timeout settings
    DATA_FETCH_TIMEOUT: int = 180  # 3 minutes
    PREPROCESSING_TIMEOUT: int = 120  # 2 minutes
    CONNECT_TIMEOUT: int = 15  # 15 seconds

    # Retry settings
    MAX_RETRIES: int = 5
    MIN_RETRY_DELAY: int = 5
    MAX_RETRY_DELAY: int = 30

    # Monitoring intervals
    PERFORMANCE_CHECK_INTERVAL: int = 24 * 60 * 60  # 24 hours
    DATA_DRIFT_CHECK_INTERVAL: int = 7 * 24 * 60 * 60  # 7 days


class Config:
    """Main configuration class."""

    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.rabbitmq = RabbitMQConfig()
        self.data_processing_service = DataProcessingServiceConfig()
        self.training_service = TrainingServiceConfig()
        self.evaluation_service = EvaluationServiceConfig()
        self.mlflow_server = MLFlowConfig()
        self.postgres = PostgresDatabaseConfig()
        self.stocks_db = StocksDatabaseConfig()
        self.news_service = NewsServiceConfig()
        self.monitoring = MonitoringConfig()

        # Create necessary directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.NEWS_DATA_DIR,
            self.preprocessing.SCALERS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Create and export global config instance
__all__ = ["config"]
config = Config()
