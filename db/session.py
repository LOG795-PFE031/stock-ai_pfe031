"""
Sets up the SQLAlchemy engine and session for PostgreSQL database interaction.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import config

# Define the URL for the database connection
DATABASE_URL = config.postgres.URL


# Create an engine that connects to the PostgreSQL database using the provided DATABASE_URL
engine = create_engine(DATABASE_URL)

# Create a session factory that will be used to create new Session objects
SessionLocal = sessionmaker(bind=engine)
