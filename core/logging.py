"""
Logging configuration for Stock AI.
"""

import logging
import sys
from typing import Dict

# Define log levels and their corresponding emojis
LOG_LEVELS = {
    "DEBUG": "🔍",
    "INFO": "✨",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
}

# Create formatters
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Create file handler
file_handler = logging.FileHandler("stock-ai.log")
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Create loggers for different components
loggers: Dict[str, logging.Logger] = {
    "main": logging.getLogger("main"),
    "rabbitmq": logging.getLogger("rabbitmq"),
    "prediction": logging.getLogger("prediction"),
    "model": logging.getLogger("model"),
    "data": logging.getLogger("data"),
    "preprocessing": logging.getLogger("preprocessing"),
    "deployment": logging.getLogger("deployment"),
    "api": logging.getLogger("api"),
    "training": logging.getLogger("training"),
    "orchestration": logging.getLogger("orchestration"),
    "news": logging.getLogger("news"),
}

# Configure each logger
for name, logger in loggers.items():
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # Prevent messages from being handled by root logger


# Add emoji to log messages
class EmojiFilter(logging.Filter):
    def filter(self, record):
        # Only add emoji if it's not already in the message
        if not any(emoji in record.getMessage() for emoji in LOG_LEVELS.values()):
            record.msg = f"{LOG_LEVELS.get(record.levelname, '')} {record.msg}"
        return True


# Add emoji filter to all handlers
emoji_filter = EmojiFilter()
console_handler.addFilter(emoji_filter)
file_handler.addFilter(emoji_filter)

# Export loggers
logger = loggers
