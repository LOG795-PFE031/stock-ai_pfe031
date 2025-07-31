"""
Database initialization script for the stock data database.
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from core.logging import logger
from .models.base import Base
from .session import STOCK_DATABASE_URL
from .models.stock_price import StockPrice  # Needs to be imported

logger = logger["data"]


async def init_stock_db():
    """Initialize the stock database by creating all tables."""
    try:
        logger.info(f"🔗 Connecting to stock database at: {STOCK_DATABASE_URL}")

        # Create a SQLAlchemy engine
        engine = create_async_engine(STOCK_DATABASE_URL)

        # Create all tables
        async with engine.begin() as conn:
            logger.info("📋 Creating database tables...")
            await conn.run_sync(Base.metadata.create_all)

        await engine.dispose()
        logger.info("✅ Stock database tables created successfully")

    except Exception as e:
        logger.error(f"❌ Error initializing stock database: {str(e)}")
        raise


if __name__ == "__main__":
    # Run this script directly to initialize the database
    asyncio.run(init_stock_db())
