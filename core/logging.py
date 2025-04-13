"""
Logging configuration for the application.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

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
    'RABBITMQ': '🐰',  # Added RabbitMQ emoji
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
    
    # Create file handler
    file_handler = FileHandlerWithFilter(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ConsoleFormatter())
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger dictionary
logger: Dict[str, logging.Logger] = {}

# List of components that need logging
components = [
    'main',
    'prediction',
    'training',
    'data',
    'api',
    'news',
    'model',
    'rabbitmq'
]

# Configure logging for each component
for component in components:
    logger[component] = setup_logging(component)
    logger[component].propagate = False  # Prevent duplicate logging 