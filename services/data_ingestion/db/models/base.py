"""
Base model for the stock data database ORM models.
"""

from sqlalchemy import Column, Integer
from sqlalchemy.ext.declarative import declared_attr

from services.data_ingestion.db.session import StockBase


class StockBaseModel(StockBase):
    """Base class for all stock data models."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True)

    @declared_attr
    def __tablename__(cls):
        """Generate __tablename__ automatically by converting the class name from CamelCase to snake_case."""
        return cls.__name__.lower()
