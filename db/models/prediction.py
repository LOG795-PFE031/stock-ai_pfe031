from sqlalchemy import Column, Integer, DECIMAL, Date, String
from .base import Base


class Prediction(Base):
    """
    SQLAlchemy model representing a stock price prediction.

    This model stores the predicted value of a stock for a given date, along with the confidence
    of the prediction and the model version used.

    Attributes:
        id (int): Primary key.
        date (date): The date the prediction is made for.
        stock_symbol (str): Ticker symbol of the stock (e.g., "AAPL").
        prediction (Decimal): The predicted stock price.
        confidence (Decimal): Confidence score of the prediction (between 0 and 1).
        model_version (int): Version number of the model used to generate the prediction.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    stock_symbol = Column(String(10), nullable=False)
    prediction = Column(DECIMAL(12, 6), nullable=False)
    confidence = Column(DECIMAL(5, 3), nullable=False)
    model_type = Column(String(50), nullable=False)
    model_version = Column(Integer, nullable=False)
