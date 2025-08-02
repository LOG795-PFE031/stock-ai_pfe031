"""
Database initialization script for the stock data database.
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from core.logging import logger
from .models.base import Base
from .session import PREDICTION_DATABASE_URL
from .models.prediction import Prediction  # Needs to be imported

logger = logger["data"]


async def init_prediction_db():
    """Initialize the prediction database by creating all tables."""
    try:
        logger.info(
            f"🔗 Connecting to the predictions database at: {PREDICTION_DATABASE_URL}"
        )

        # Create a SQLAlchemy engine
        engine = create_async_engine(PREDICTION_DATABASE_URL)

        # Create all tables
        async with engine.begin() as conn:
            logger.info("📋 Creating database tables...")
            await conn.run_sync(Base.metadata.create_all)

        await engine.dispose()
        logger.info("✅ Prediction database tables created successfully")

    except Exception as e:
        logger.error(f"❌ Error initializing prediction database: {str(e)}")
        raise


if __name__ == "__main__":
    # Run this script directly to initialize the database
    asyncio.run(init_prediction_db())
