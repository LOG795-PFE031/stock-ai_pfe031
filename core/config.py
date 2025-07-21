"""
Configuration settings for the Stock AI system.
"""

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


class PostgresDatabaseConfig(BaseModel):
    """PostgreSQL configuration"""

    HOST: str = "postgres-stock-ai"
    PORT: int = 5432
    USER: str = "admin"
    PASSWORD: str = "admin"

    @property
    def URL(self) -> str:
        return (
            f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/stocks"
        )


class Config:
    """Main configuration class."""

    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.rabbitmq = RabbitMQConfig()
        self.postgres = PostgresDatabaseConfig()

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
