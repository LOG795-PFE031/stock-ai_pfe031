import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas_market_calendars as mcal
from pytz import timezone
from sqlalchemy import select
import httpx

from core.logging import logger
from core.utils import format_prediction_response, get_next_trading_day
from core.config import config
from core import BaseService
from .db.models.prediction import Prediction
from .db.session import get_prediction_async_session

from .prediction_storage import PredictionStorage
from .flows import (
    run_evaluation_flow,
    run_prediction_flow,
    run_training_flow,
    run_batch_prediction,
    run_historical_predictions_flow,
)


class OrchestrationService(BaseService):
    """
    OrchestrationService manages the full lifecycle of ML workflows, including training,
    prediction, evaluation, and historical analysis. It integrates multiple services and
    handles scheduling for automated batch predictions.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.logger = logger["orchestration"]
        self.prediction_storage = PredictionStorage(self.logger)

    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        try:
            # Configure the prediction scheduler
            self._set_prediction_scheduler()

            # Test connection to the prediction database
            try:
                AsyncSessionLocal = get_prediction_async_session()
                async with AsyncSessionLocal() as session:
                    # Simple query to test database connection and create tables if needed
                    try:
                        stmt = select(Prediction).limit(1)
                        await session.execute(stmt)
                        self.logger.info(
                            "✅ Connection to the prediction database verified successfully"
                        )
                    except Exception as table_error:
                        # Table might not exist yet, try to create it
                        self.logger.warning(
                            f"⚠️ Prediction table not found, attempting to create: {table_error}"
                        )
                        try:
                            from .db.init_db import init_prediction_db

                            await init_prediction_db()
                            self.logger.info("✅ Database tables created successfully")
                        except Exception as create_error:
                            self.logger.error(
                                f"❌ Failed to create database tables: {create_error}"
                            )
                            raise
            except Exception as db_error:
                self.logger.error(
                    f"❌ Error connecting to prediction database: {str(db_error)}"
                )
                raise

            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize orchestration service: %s", str(e))
            raise

    async def run_training_pipeline(self, model_type: str, symbol: str):
        """
        Run the full training pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being trained.

        Returns:
            result: The result of the training pipeline execution.
        """

        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                "Starting training pipeline for %s model for %s.", model_type, symbol
            )

            # Run the training pipeline in a background thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                run_training_flow,
                model_type,
                symbol,
            )

            self.logger.info(
                "Training pipeline completed successfully for %s model for %s.",
                model_type,
                symbol,
            )

            # Add a success status to the result
            result["status"] = "success"

            return result
        except Exception as e:
            self.logger.error(
                "Error running the training pipeline for model %s for %s: %s",
                model_type,
                symbol,
                str(e),
            )

            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_prediction_pipeline(self, model_type: str, symbol: str):
        """
        Run the full prediction pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being used for prediction.

        Returns:
            result: The result of the prediction pipeline execution.
        """
        try:
            symbol = symbol.upper()

            self.logger.info(
                "Starting prediction pipeline for %s model for %s.", model_type, symbol
            )

            # Get the predicted price date
            next_trading_day = get_next_trading_day()

            result = await self.prediction_storage.load_prediction_from_db(
                model_type=model_type, symbol=symbol, date=next_trading_day
            )

            if result is not None:
                self.logger.info(
                    "Serving prediction from cache for %s model for %s.",
                    model_type,
                    symbol,
                )

                return format_prediction_response(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    model_type=model_type,
                    symbol=symbol,
                    model_version=result["model_version"],
                    date=next_trading_day,
                )

            self.logger.info(
                "Computing the prediction for %s model for %s...", model_type, symbol
            )

            # Run the prediction pipeline
            prediction_result = await asyncio.to_thread(
                run_prediction_flow,
                model_type,
                symbol,
            )

            if prediction_result:

                # Extract the prediction results
                prediction = prediction_result["y_pred"][0]
                confidence = prediction_result["confidence"][0]
                model_version = prediction_result["model_version"]

                self.logger.info(
                    "Prediction pipeline completed for %s model for %s with version %s.",
                    model_type,
                    symbol,
                    model_version,
                )

                # Save prediction to db
                await self.prediction_storage.save_prediction_to_db(
                    model_type=model_type,
                    symbol=symbol,
                    date=next_trading_day,
                    prediction=prediction,
                    confidence=confidence,
                    model_version=model_version,
                )

                return format_prediction_response(
                    prediction=prediction,
                    confidence=confidence,
                    model_type=model_type,
                    symbol=symbol,
                    model_version=model_version,
                    date=next_trading_day,
                )

            self.logger.info(
                "No live model available to make prediction with %s model for %s.",
                model_type,
                symbol,
            )

            return {
                "status": "error",
                "error": f"No live model available to make prediction with {model_type} "
                + "model for {symbol}.",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(
                "Error running the prediction pipeline for model %s for %s: %s",
                model_type,
                symbol,
                str(e),
            )

            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_evaluation_pipeline(self, model_type: str, symbol: str):
        """
        Run the full evaluation pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being evaluated.

        Returns:
            result: The result of the evaluation pipeline execution.
        """

        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                "Starting evaluation pipeline for %s model for %s.", model_type, symbol
            )

            # Run the evaluation pipeline
            result = await asyncio.to_thread(
                run_evaluation_flow,
                model_type,
                symbol,
            )

            # Log the successful completion of the pipeline
            self.logger.info(
                "Evaluation pipeline completed successfully for %s model for %s.",
                model_type,
                symbol,
            )

            return result
        except Exception as e:
            self.logger.error(
                "Error running the evaluation pipeline for model %s for %s: %s",
                model_type,
                symbol,
                str(e),
            )

            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_historical_prediction_pipeline(
        self, model_type: str, symbol: str, start_date: datetime, end_date: datetime
    ):
        """
        Run the historical prediction pipeline for a given stock symbol and model type over a
            specified date range.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being evaluated.
            start_date (datetime): The start of the historical prediction range.
            end_date (datetime): The end of the historical prediction range.

        Returns:
            dict: The result of the historical prediction pipeline execution.
        """
        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                "Starting historical prediction for %s model for %s from %s to %s.",
                model_type,
                symbol,
                start_date,
                end_date,
            )

            # Get trading dates range
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index.to_pydatetime().tolist()

            # Get the iso format of the trading days
            target_dates = [dt.date().isoformat() for dt in trading_days]

            # Get the existing dates (in the db)
            existing_dates = (
                await self.prediction_storage.get_existing_prediction_dates(
                    model_type=model_type, symbol=symbol
                )
            )

            # Check if any date in the target range is missing
            missing_dates = [
                dt
                for dt, dt_str in zip(trading_days, target_dates)
                if dt_str not in existing_dates
            ]

            if missing_dates:
                self.logger.info(
                    "Computing predictions for %s model for %s...", model_type, symbol
                )

                # Makes predictions
                predictions = await asyncio.to_thread(
                    run_historical_predictions_flow,
                    model_type,
                    symbol,
                    trading_days,
                )

                if predictions:

                    # Initialize the results list
                    results = []

                    for i in range(len(predictions)):

                        # Extract the prediction results
                        prediction = predictions[i]["y_pred"][0]
                        confidence = predictions[i]["confidence"][0]
                        model_version = predictions[i]["model_version"]

                        # Save prediction to db
                        await self.prediction_storage.save_prediction_to_db(
                            model_type=model_type,
                            symbol=symbol,
                            date=trading_days[i],
                            prediction=prediction,
                            confidence=confidence,
                            model_version=model_version,
                        )

                        # Add formated prediction response to the results list
                        results.append(
                            format_prediction_response(
                                prediction=prediction,
                                confidence=confidence,
                                model_type=model_type,
                                symbol=symbol,
                                model_version=model_version,
                                date=trading_days[i],
                            )
                        )

                else:
                    self.logger.info(
                        "No live model available to make predictions with %s model for %s.",
                        model_type,
                        symbol,
                    )

                    return {
                        "status": "error",
                        "error": f"No live model available to make predictions with {model_type} "
                        + "model for {symbol}.",
                        "symbol": symbol,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                self.logger.info(
                    "Serving predictions from cache for %s model for %s.",
                    model_type,
                    symbol,
                )

                results = []
                for date in trading_days:
                    prediction_result = (
                        await self.prediction_storage.load_prediction_from_db(
                            model_type=model_type, symbol=symbol, date=date
                        )
                    )

                    results.append(
                        format_prediction_response(
                            prediction=prediction_result["prediction"],
                            confidence=prediction_result["confidence"],
                            model_type=model_type,
                            symbol=symbol,
                            model_version=prediction_result["model_version"],
                            date=date,
                        )
                    )

            return {
                "symbol": symbol,
                "predictions": results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(
                "Error running the historical prediction pipeline for model %s for %s: %s",
                model_type,
                symbol,
                str(e),
            )

            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Orchestration service cleaned up successfully")
        except Exception as e:
            self.logger.error("Error during orchestration service cleanup: %s", str(e))

    def _set_prediction_scheduler(self):
        """Set the prediction scheduler"""

        # Create and start the scheduler
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()

        # Configure a trigger for the scheduler (15 mins after the market close on weekdays)
        eastern = timezone("US/Eastern")
        trigger = CronTrigger(
            day_of_week="mon-fri", hour=16, minute=45, timezone=eastern
        )

        # Add job to the scheduler
        self.scheduler.add_job(self._predict_all, trigger=trigger)

    async def _get_trainers(self):
        """Call the training service to get all the available trainers"""
        url = f"http://{config.training_service.HOST}:{config.training_service.PORT}/training/trainers"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)

            # Check if the response is successful
            response.raise_for_status()
            return response.json()

    async def _predict_all(self):
        """Run batch prediction for all configured model types and symbols."""

        # Retrieve the model types
        trainers = await self._get_trainers()
        model_types = trainers["types"]

        # Retrieve the symbols to predict
        data_service_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stocks"

        async with httpx.AsyncClient() as client:
            response = await client.get(data_service_url)
            response.raise_for_status()
            stocks_data = response.json()

        symbols = [item["symbol"] for item in stocks_data["data"]]

        run_batch_prediction(
            model_types,
            symbols,
        )
