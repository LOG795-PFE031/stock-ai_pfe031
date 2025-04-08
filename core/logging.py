"""
Logging configuration for the application.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import config

# Emoji indicators for different log levels
EMOJI_INDICATORS = {
    'DEBUG': '🔍',
    'INFO': '✨',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '🚨',
    # Task-specific emojis
    'MODEL_LOAD': '🤖',
    'PREDICTION': '🎯',
    'TRAINING': '📈',
    'DATA': '📊',
    'API': '🌐',
    'NEWS': '📰',
}

class ConsoleFormatter(logging.Formatter):
    """Custom formatter for console output with emojis."""
    
    def format(self, record):
        # Get the appropriate emoji
        if hasattr(record, 'emoji'):
            emoji = record.emoji
        else:
            emoji = EMOJI_INDICATORS.get(record.levelname, '✨')
        
        # For model loading and predictions, add specific emojis
        if 'model' in record.msg.lower():
            emoji = EMOJI_INDICATORS['MODEL_LOAD']
        elif 'predict' in record.msg.lower():
            emoji = EMOJI_INDICATORS['PREDICTION']
        
        # Format the message
        if record.levelno == logging.INFO:
            # Simpler format for INFO level
            return f"{emoji} {record.getMessage()}"
        else:
            # More detailed format for other levels
            return f"{emoji} [{record.levelname}] {record.getMessage()}"

class NonEmptyFilter(logging.Filter):
    """Filter to prevent empty log files."""
    
    def filter(self, record):
        """Only allow non-empty messages."""
        return bool(record.getMessage().strip())

class FileHandlerWithFilter(logging.FileHandler):
    """File handler that only creates files when there's content."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addFilter(NonEmptyFilter())
        self.has_content = False
    
    def emit(self, record):
        """Emit a record and mark that we have content."""
        if self.filter(record):
            self.has_content = True
            super().emit(record)
    
    def close(self):
        """Close the handler and remove empty files."""
        super().close()
        if not self.has_content and os.path.exists(self.baseFilename):
            try:
                os.remove(self.baseFilename)
            except OSError:
                pass

def setup_logging(component: str) -> logging.Logger:
    """
    Setup logging for a specific component.
    
    Args:
        component: Name of the component (e.g., 'training', 'prediction', 'api')
        
    Returns:
        Logger instance for the component
    """
    # Create logger
    logger = logging.getLogger(component)
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    log_dir = config.data.LOGS_DIR / component
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{component}_{timestamp}.log"
    
    # Create file handler with filter
    file_handler = FileHandlerWithFilter(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = ConsoleFormatter()
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logging
    logger.propagate = False
    
    return logger

# Create loggers for different components
training_logger = setup_logging('training')
prediction_logger = setup_logging('prediction')
api_logger = setup_logging('api')
data_logger = setup_logging('data')
model_logger = setup_logging('model')
news_logger = setup_logging('news')
main_logger = setup_logging('main')

# Export loggers
logger = {
    'training': training_logger,
    'prediction': prediction_logger,
    'api': api_logger,
    'data': data_logger,
    'model': model_logger,
    'news': news_logger,
    'main': main_logger
}

# Configure logging for each component
for name in logger:
    logger[name] = logging.getLogger(name)
    logger[name].setLevel(logging.INFO)
    
    # Add console handler if not already present
    if not logger[name].handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger[name].addHandler(console_handler)
        
        # Add file handler
        file_handler = logging.FileHandler(config.data.LOGS_DIR / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger[name].addHandler(file_handler)
    
    logger[name].propagate = False  # Prevent duplicate logging 