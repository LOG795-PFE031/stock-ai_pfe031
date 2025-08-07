"""
Sets up the SQLAlchemy engine and session for MySQL database interaction.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import config

# Define the URL for the stock database connection (sync)
DATABASE_URL = config.mysql.URL_sync

# Create the engine ONCE at module level (sync version)
engine = create_engine(
    DATABASE_URL,
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


def get_sync_session():
    """Return a new sync SQLAlchemy session for the stock database."""
    return SessionLocal
