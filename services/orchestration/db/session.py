"""
Sets up the SQLAlchemy engine and session for the dedicated prediction data PostgreSQL database.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from core.config import config

# Define the URL for the stock database connection
PREDICTION_DATABASE_URL = config.predictions_db.URL


def get_prediction_async_session():
    """
    Create a session factory that will be used to create new async Session objects for the
    prediction database.
    """
    engine = create_async_engine(PREDICTION_DATABASE_URL)
    return async_sessionmaker(bind=engine)
