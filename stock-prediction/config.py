"""
Configuration module for stock prediction models.
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    """Data configuration"""
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.1
    SEQUENCE_LENGTH: int = 60
    FEATURE_COLUMNS: list = None
    
    def __post_init__(self):
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'RSI', 'MACD', 'MACD_Signal', 'Volatility',
                'MA_5', 'MA_20', 'MA_50', 'MA_200',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'Volume_MA', 'Volume_Ratio', 'Momentum',
                'ATR', 'Stoch_K', 'Stoch_D'
            ]

@dataclass
class ModelConfig:
    """Model configuration"""
    MODEL_DIR: str = "models"
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.2
    LSTM_UNITS: int = 50
    DENSE_UNITS: int = 25
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-6
    MAX_LR: float = 1e-3
    WARMUP_EPOCHS: int = 5
    COOLDOWN_EPOCHS: int = 0
    USE_GPU: bool = True
    GPU_MEMORY_LIMIT: Optional[int] = None
    GPU_MEMORY_GROWTH: bool = True

@dataclass
class ProphetConfig:
    """Prophet model configuration"""
    CHANGEPOINT_PRIOR_SCALE: float = 0.05
    HOLIDAYS_PRIOR_SCALE: float = 10.0
    SEASONALITY_PRIOR_SCALE: float = 10.0
    SEASONALITY_MODE: str = "multiplicative"
    CHANGEPOINT_RANGE: float = 0.8
    INTERVAL_WIDTH: float = 0.95
    STAN_BACKEND: str = "cmdstanpy"
    MCMC_SAMPLES: int = 0
    N_CHANGEPOINTS: int = 25
    CHANGEPOINT_PRIOR_SCALE: float = 0.05
    HOLIDAYS_PRIOR_SCALE: float = 10.0
    SEASONALITY_PRIOR_SCALE: float = 10.0
    SEASONALITY_MODE: str = "multiplicative"
    CHANGEPOINT_RANGE: float = 0.8
    INTERVAL_WIDTH: float = 0.95
    STAN_BACKEND: str = "cmdstanpy"
    MCMC_SAMPLES: int = 0
    N_CHANGEPOINTS: int = 25

@dataclass
class LoggingConfig:
    """Logging configuration"""
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    prophet: ProphetConfig = ProphetConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.data.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.model.MODEL_DIR, exist_ok=True)
        os.makedirs(self.logging.LOG_DIR, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config instance from a dictionary
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Config instance
        """
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        prophet_config = ProphetConfig(**config_dict.get('prophet', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            prophet=prophet_config,
            logging=logging_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config instance to dictionary
        
        Returns:
            Dictionary containing configuration parameters
        """
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'prophet': self.prophet.__dict__,
            'logging': self.logging.__dict__
        } 