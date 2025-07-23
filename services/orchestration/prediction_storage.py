from datetime import datetime

import pandas as pd
from sqlalchemy import select

from db.session import get_async_session
from db.models.prediction import Prediction


class PredictionStorage:
    """
    Handles saving, loading, and retrieving stock prediction data
    from a PostgreSQL database using SQLAlchemy.

    This class provides methods to:
    - Save predictions made by different models (e.g., LSTM, Prophet).
    - Load a prediction for a given stock, model type, and date.
    - Retrieve all dates for which predictions exist for a given stock and model.

    Attributes:
        logger (logging.Logger): Logger instance for tracking operations and errors.
    """

    def __init__(self, logger):
        self.logger = logger

    async def load_prediction_from_db(
        self, model_type: str, symbol: str, date: datetime
    ):
        """
        Load prediction results from the postgres db for a specified date.

        Args:
            model_type (str): The type of model used for prediction (e.g., LSTM, Prophet).
            symbol (str): The stock symbol being predicted.
            date (datetime): Date associated to the prediction

        Returns:
            dict|None: The prediction results if Any, else None
        """

        # Create a new SQLAlchemy session to interact with the database
        AsyncSessionLocal = get_async_session()
        async with AsyncSessionLocal() as session:
            try:
                # Check if there is a prediction given the symbol, the date and the model_type
                stmt = select(Prediction).filter(
                    Prediction.stock_symbol == symbol,
                    Prediction.date == date.date(),
                    Prediction.model_type == model_type,
                )
                result = await session.execute(stmt)

                # Retrieve the first matching prediction (if any)
                prediction = result.scalars().first()

                if prediction:
                    return {
                        "date": prediction.date.isoformat(),
                        "stock_symbol": prediction.stock_symbol,
                        "prediction": float(prediction.prediction),
                        "confidence": float(prediction.confidence),
                        "model_type": prediction.model_type,
                        "model_version": prediction.model_version,
                    }

                # If no prediction
                return None

            except Exception as e:
                self.logger.error("Error loading prediction for %s: %s", symbol, str(e))
                raise e

    async def get_existing_prediction_dates(
        self, model_type: str, symbol: str
    ) -> list[str]:
        """
        Get a list of existing dates for which predictions have already been computed and stored
        in the postgres db.

        Args:
            model_type (str): Model type (e.g., "lstm", "prophet").
            symbol (str): Stock symbol (e.g., "AAPL").

        Returns:
            list[str]: List of dates (in 'YYYY-MM-DD' format) that already have predictions.
        """

        # Create a new SQLAlchemy session to interact with the database
        AsyncSessionLocal = get_async_session()
        async with AsyncSessionLocal() as session:
            try:

                # Query predictions for the given symbol
                query = select(Prediction.date).where(
                    Prediction.stock_symbol == symbol,
                    Prediction.model_type == model_type,
                )
                result = await session.execute(query)
                dates = result.scalars().all()

                if not dates:
                    return []

                # Convert dates to strings directly (no DataFrame needed)
                return [date.strftime("%Y-%m-%d") for date in dates]

            except Exception as e:
                self.logger.error(
                    "Error fetching existing prediction dates for symbol %s: %s",
                    symbol,
                    str(e),
                )
                raise e

    async def save_prediction_to_db(
        self,
        model_type: str,
        symbol: str,
        date: datetime,
        prediction: float,
        confidence: float,
        model_version: str,
    ):
        """
        Save the prediction results to the postgres db.

        Args:
            model_type (str): The type of model used for prediction (e.g., LSTM, Prophet).
            symbol (str): The stock symbol being predicted.
            date (datetime): Date associated to the prediction
            prediction (float): The predicted value from the model.
            confidence (float): The confidence score of the prediction.
            model_version (str): The version of the model used.
        """

        # Create a new SQLAlchemy session to interact with the database
        AsyncSessionLocal = get_async_session()
        async with AsyncSessionLocal() as session:
            try:
                # Check if prediction already exists for that symbol and date
                query = select(Prediction).where(
                    Prediction.stock_symbol == symbol,
                    Prediction.date == date.date(),
                    Prediction.model_type == model_type,
                )
                result = await session.execute(query)
                existing = result.scalars().first()

                if existing:
                    # Update
                    existing.prediction = float(prediction)
                    existing.confidence = float(confidence)
                    existing.model_version = model_version

                else:
                    # Insert
                    new_prediction = Prediction(
                        date=date.date(),
                        stock_symbol=symbol,
                        prediction=float(prediction),
                        confidence=float(confidence),
                        model_version=model_version,
                        model_type=model_type,
                    )
                    session.add(new_prediction)

                # Commit the transaction
                await session.commit()

                self.logger.debug(
                    "Saved prediction for %s on %s using model {model_type}",
                    symbol,
                    date.date(),
                )

            except Exception as e:
                self.logger.error("Error saving prediction for %s: %s", symbol, str(e))
                raise e
