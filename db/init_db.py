from sqlalchemy import create_engine
from core.config import config
from .models.base import Base
from .models.stock_price import StockPrice
from .models.prediction import Prediction


def create_database():
    """
    Create database tables using SQLAlchemy models defined in the application.

    Connects to the PostgreSQL database using the configured URL, and creates
    all tables registered in the `Base` metadata (StockPrice and Prediction).
    """

    database_url = config.postgres.URL

    # Use sync engine for table creation
    engine = create_engine(database_url.replace("postgresql+asyncpg", "postgresql"))
    Base.metadata.create_all(engine)
    print("Stock price table and predictions table created successfully!")


# Call this function to initialize the database.
if __name__ == "__main__":
    create_database()
