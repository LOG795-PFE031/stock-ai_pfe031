"""
Sets up the SQLAlchemy engine and session for the dedicated stock data PostgreSQL database (sync version).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import config

# Define the URL for the stock database connection (sync)
STOCK_DATABASE_URL = config.stocks_db.URL_sync

# Create the engine ONCE at module level (sync version)
engine = create_engine(
    STOCK_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=10,
)

# Create sessionmaker
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_stock_session():
    """Return a new sync SQLAlchemy session for the stock database."""
    return SessionLocal
