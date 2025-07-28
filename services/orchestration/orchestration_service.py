import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas_market_calendars as mcal
from pytz import timezone
import httpx

from core.logging import logger
from core.utils import format_prediction_response, get_next_trading_day
from core.config import config
from core import BaseService
from services.deployment import DeploymentService

from services.evaluation import EvaluationService
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
        deployment_service: DeploymentService,
        evaluation_service: EvaluationService,
    ):
        super().__init__()
        self.deployment_service = deployment_service
        self.evaluation_service = evaluation_service
        self.logger = logger["orchestration"]
        self.prediction_storage = PredictionStorage(self.logger)

    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        try:
            # Configure the prediction scheduler
            self._set_prediction_scheduler()

            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize orchestration service: %s", str(e))
            raise

    async def _fetch_stock_data(self, symbol: str, days_back: int = None):
        """
        Fetch stock data from the API endpoint.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days_back: Number of days to look back (optional, defaults to LOOKBACK_PERIOD_DAYS)
            
        Returns:
            Dictionary containing stock data
        """
        try:
            # API endpoint URL
            api_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stock/recent"
            params = {"symbol": symbol}
            
            # Use default lookback period if days_back is not specified
            if days_back is None:
                days_back = config.data.LOOKBACK_PERIOD_DAYS
            
            params["days_back"] = days_back
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(api_url, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise

    async def _fetch_nasdaq_stocks(self):
        """
        Fetch list of NASDAQ stocks from the API endpoint.
        
        Returns:
            Dictionary containing list of stocks
        """
        try:
            # API endpoint URL for stocks list
            api_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stocks"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(api_url)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            self.logger.error(f"Error fetching NASDAQ stocks: {e}")
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
                self._fetch_stock_data,
                self.deployment_service,
                self.evaluation_service,
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
            # Enforce upper case for the symbol
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
                self._fetch_stock_data,
                self.deployment_service,
            )

            if prediction_result["status"] == "success":
                # Store the prediction in the database
                await self.prediction_storage.store_prediction_in_db(
                    model_type=model_type,
                    symbol=symbol,
                    prediction=prediction_result["prediction"],
                    confidence=prediction_result["confidence"],
                    model_version=prediction_result["model_version"],
                    date=next_trading_day,
                )

                return format_prediction_response(
                    prediction=prediction_result["prediction"],
                    confidence=prediction_result["confidence"],
                    model_type=model_type,
                    symbol=symbol,
                    model_version=prediction_result["model_version"],
                    date=next_trading_day,
                )
            else:
                return prediction_result

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

            # Run the evaluation pipeline in a background thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                run_evaluation_flow,
                model_type,
                symbol,
                self._fetch_stock_data,
                self.deployment_service,
                self.evaluation_service,
            )

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
        Run the full historical prediction pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being used for prediction.
            start_date (datetime): Start date for historical predictions.
            end_date (datetime): End date for historical predictions.

        Returns:
            result: The result of the historical prediction pipeline execution.
        """
        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                "Starting historical prediction pipeline for %s model for %s.",
                model_type,
                symbol,
            )

            # Get the trading days between start_date and end_date
            nyse = mcal.get_calendar("NYSE")
            trading_days = nyse.schedule(
                start_date=start_date, end_date=end_date
            ).index.tolist()

            # Check if we have cached predictions for all trading days
            cached_predictions = []
            missing_dates = []

            for date in trading_days:
                prediction_result = (
                    await self.prediction_storage.load_prediction_from_db(
                        model_type=model_type, symbol=symbol, date=date
                    )
                )

                if prediction_result is not None:
                    cached_predictions.append(
                        format_prediction_response(
                            prediction=prediction_result["prediction"],
                            confidence=prediction_result["confidence"],
                            model_type=model_type,
                            symbol=symbol,
                            model_version=prediction_result["model_version"],
                            date=date,
                        )
                    )
                else:
                    missing_dates.append(date)

            # If we have missing predictions, compute them
            if missing_dates:
                self.logger.info(
                    "Computing historical predictions for %s model for %s...",
                    model_type,
                    symbol,
                )

                # Run the historical prediction pipeline
                historical_predictions = await asyncio.to_thread(
                    run_historical_predictions_flow,
                    model_type,
                    symbol,
                    missing_dates,
                    self._fetch_stock_data,
                    self.deployment_service,
                )

                if historical_predictions["status"] == "success":
                    # Store the predictions in the database
                    for prediction in historical_predictions["predictions"]:
                        await self.prediction_storage.store_prediction_in_db(
                            model_type=model_type,
                            symbol=symbol,
                            prediction=prediction["prediction"],
                            confidence=prediction["confidence"],
                            model_version=prediction["model_version"],
                            date=prediction["date"],
                        )

                    # Combine cached and new predictions
                    results = cached_predictions + historical_predictions["predictions"]
                else:
                    return historical_predictions
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
        stocks_data = await self._fetch_nasdaq_stocks()
        symbols = [item["symbol"] for item in stocks_data["data"]]

        run_batch_prediction(
            model_types,
            symbols,
            self._fetch_stock_data,
            self.deployment_service,
        )
