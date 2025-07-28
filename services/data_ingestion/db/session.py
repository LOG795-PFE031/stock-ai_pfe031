"""
Sets up the SQLAlchemy engine and session for the dedicated stock data PostgreSQL database.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from core.config import config

# Define the URL for the stock database connection
STOCK_DATABASE_URL = config.stocks_db.URL


def get_stock_async_session():
    """Create a session factory that will be used to create new async Session objects for the stock database"""
    engine = create_async_engine(STOCK_DATABASE_URL)
    return async_sessionmaker(bind=engine)
