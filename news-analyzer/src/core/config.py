from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # RabbitMQ configuration
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"

    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8092

    # Rate limiting
    RATE_LIMIT: int = 2  # requests per period
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # Model configuration
    MODEL_PATH: Optional[str] = None

    # Optional News API key (not required as we use yfinance)
    NEWS_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True 