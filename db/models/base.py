"""
Defines the SQLAlchemy declarative base class used for all ORM models.

This base class (`Base`) is imported by each model in the project to ensure
they share the same metadata and are properly registered with SQLAlchemy's ORM.

All model classes should inherit from this `Base` to be included in the
database schema generation and ORM functionality.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
