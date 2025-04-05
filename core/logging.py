"""
Logging configuration for the application.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import config

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
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
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