from sqlalchemy import Column, String, Integer, Numeric, Date
from .base import Base


class StockPrice(Base):
    """
    SQLAlchemy model representing historical stock price data.

    This model stores daily stock price information for a specific stock symbol,
    including OHLC (Open, High, Low, Close) data, volume, dividends, and stock splits.

    Attributes:
        id (int): Primary key.
        stock_symbol (str): Ticker symbol of the stock (e.g., "AAPL").
        stock_name (str): Optional human-readable name of the stock (e.g., "Apple Inc.").
        date (date): The date of the stock price record (in YYYY-MM-DD format).
        open (Decimal): Opening price of the stock on the given date.
        high (Decimal): Highest price reached during the trading day.
        low (Decimal): Lowest price during the trading day.
        close (Decimal): Closing price of the stock for the day.
        volume (int): Number of shares traded.
        dividends (Decimal): Cash dividend paid per share on the given date.
        stock_splits (Decimal): Ratio of any stock split on the given date.
    """

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True)
    stock_symbol = Column(String(10), nullable=False)
    stock_name = Column(String(100))
    date = Column(Date, nullable=False)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4))
    volume = Column(Integer)
    dividends = Column(Numeric(12, 4))
    stock_splits = Column(Numeric(12, 4))
