"""
Sets up the SQLAlchemy engine and session for PostgreSQL database interaction.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from core.config import config

# Define the URL for the database connection
DATABASE_URL = config.postgres.URL


def get_async_session():
    """Create a session factory that will be used to create new async Session objects"""
    engine = create_async_engine(DATABASE_URL)
    return async_sessionmaker(bind=engine)
