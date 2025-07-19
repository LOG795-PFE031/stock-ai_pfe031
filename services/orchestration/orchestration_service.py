from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from asyncio import run
from datetime import datetime
import pandas_market_calendars as mcal
from pytz import timezone

from core.logging import logger
from core.utils import format_prediction_response, get_next_trading_day
from .prediction_storage import PredictionStorage
from .flows import (
    run_evaluation_flow,
    run_prediction_flow,
    run_training_flow,
    run_batch_prediction,
    run_historical_predictions_flow,
)
from ..base_service import BaseService
from ..deployment import DeploymentService
from ..training import TrainingService
from ..data_processing import DataProcessingService
from ..evaluation_service import EvaluationService
from ..data_service import DataService
from core.logging import logger
from core.config import config


class OrchestrationService(BaseService):

    def __init__(
        self,
        data_service: DataService,
        data_processing_service: DataProcessingService,
        training_service: TrainingService,
        deployment_service: DeploymentService,
        evaluation_service: EvaluationService,
    ):
        super().__init__()
        self.data_service = data_service
        self.data_processing_service = data_processing_service
        self.training_service = training_service
        self.deployment_service = deployment_service
        self.evaluation_service = evaluation_service
        self.logger = logger["orchestration"]
        self.prediction_storage = PredictionStorage(self.logger)

        # Scheduler (use to schdule all predictions)
        self.scheduler = AsyncIOScheduler()

    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        try:
            # Configure the prediction scheduler
            self._set_prediction_scheduler()

            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration service: {str(e)}")
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
                f"Starting training pipeline for {model_type} model for {symbol}."
            )

            # Run the training pipeline
            result = run_training_flow(
                model_type,
                symbol,
                self.data_service,
                self.data_processing_service,
                self.training_service,
                self.deployment_service,
                self.evaluation_service,
            )

            self.logger.info(
                f"Training pipeline completed successfully for {model_type} model for {symbol}."
            )

            # Add a success status to the result
            result["status"] = "success"

            return result
        except Exception as e:
            self.logger.error(
                f"Error running the training pipeline for model {model_type} for {symbol}: {str(e)}"
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
                f"Starting prediction pipeline for {model_type} model for {symbol}."
            )

            # Get the predicted price date
            next_trading_day = get_next_trading_day()

            result = self.prediction_storage.load_prediction_csv(
                model_type=model_type, symbol=symbol, date=next_trading_day
            )

            if result is not None:
                self.logger.info(
                    f"Serving prediction from cache for {model_type} model for {symbol}."
                )
                return format_prediction_response(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    model_type=result["model_type"],
                    symbol=result["symbol"],
                    model_version=result["model_version"],
                    date=next_trading_day,
                )

            self.logger.info(
                f"Computing the prediction for {model_type} model for {symbol}..."
            )

            # Run the prediction pipeline
            prediction_result = run_prediction_flow(
                model_type,
                symbol,
                self.data_service,
                self.data_processing_service,
                self.deployment_service,
            )

            if prediction_result:

                # Extract the prediction results
                prediction = prediction_result["y_pred"][0]
                confidence = prediction_result["confidence"][0]
                model_version = prediction_result["model_version"]

                self.logger.info(
                    f"Prediction pipeline completed successfully for {model_type} model for {symbol}."
                )

                # Save prediction to csv
                self.prediction_storage.save_prediction_to_csv(
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
            else:
                self.logger.info(
                    f"No live model available to make prediction with {model_type} model for {symbol}."
                )
                return {
                    "status": "error",
                    "error": f"No live model available to make prediction with {model_type} model for {symbol}.",
                    "symbol": symbol,
                    "model_type": model_type,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(
                f"Error running the prediction pipeline for model {model_type} for {symbol}: {str(e)}"
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
                f"Starting evaluation pipeline for {model_type} model for {symbol}."
            )

            # Run the evaluation pipeline
            result = run_evaluation_flow(
                model_type,
                symbol,
                self.data_service,
                self.data_processing_service,
                self.deployment_service,
                self.evaluation_service,
            )

            # Log the successful completion of the pipeline
            self.logger.info(
                f"Evaluation pipeline completed successfully for {model_type} model for {symbol}."
            )

            return result
        except Exception as e:
            self.logger.error(
                f"Error running the evaluation pipeline for model {model_type} for {symbol}: {str(e)}"
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
        Run the historical prediction pipeline for a given stock symbol and model type over a specified date range.

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
                f"Starting historical prediction for {model_type} model for {symbol} from {start_date} to {end_date}."
            )

            # Get trading dates range
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index.to_pydatetime().tolist()

            # Get the iso format of the trading days
            target_dates = [dt.date().isoformat() for dt in trading_days]

            # Get the existing dates in the csv file (db)
            existing_dates = self.prediction_storage.get_existing_prediction_dates(
                model_type=model_type, symbol=symbol
            )

            # Check if any date in the target range is missing
            missing_dates = [
                dt
                for dt, dt_str in zip(trading_days, target_dates)
                if dt_str not in existing_dates
            ]

            if missing_dates:
                self.logger.info(
                    f"Computing predictions for {model_type} model for {symbol}..."
                )

                # Makes predictions
                predictions = run_historical_predictions_flow(
                    model_type=model_type,
                    symbol=symbol,
                    trading_days=trading_days,
                    data_service=self.data_service,
                    processing_service=self.data_processing_service,
                    deployment_service=self.deployment_service,
                )

                if predictions:

                    # Initialize the results list
                    results = []

                    for i in range(len(predictions)):

                        # Extract the prediction results
                        prediction = predictions[i]["y_pred"][0]
                        confidence = predictions[i]["confidence"][0]
                        model_version = predictions[i]["model_version"]

                        # Get formatted result
                        formatted_result = format_prediction_response(
                            prediction=prediction,
                            confidence=confidence,
                            model_type=model_type,
                            symbol=symbol,
                            model_version=model_version,
                            date=trading_days[i],
                        )

                        # Save prediction to csv
                        self.prediction_storage.save_prediction_to_csv(
                            model_type=model_type,
                            symbol=symbol,
                            date=formatted_result["date"],
                            prediction=prediction,
                            confidence=confidence,
                            model_version=model_version,
                        )

                        # Add formated prediction response to the results list
                        results.append(formatted_result)

                else:
                    self.logger.info(
                        f"No live model available to make predictions with {model_type} model for {symbol}."
                    )
                    return {
                        "status": "error",
                        "error": f"No live model available to make predictions with {model_type} model for {symbol}.",
                        "symbol": symbol,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                self.logger.info(
                    f"Serving predictions from cache for {model_type} model for {symbol}."
                )
                results = []
                for date in trading_days:
                    prediction_result = self.prediction_storage.load_prediction_csv(
                        model_type=model_type, symbol=symbol, date=date
                    )

                    results.append(
                        format_prediction_response(
                            prediction=prediction_result["prediction"],
                            confidence=prediction_result["confidence"],
                            model_type=prediction_result["model_type"],
                            symbol=prediction_result["symbol"],
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
                f"Error running the prediction pipeline for model {model_type} for {symbol}: {str(e)}"
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
            self.logger.error(f"Error during orchestration service cleanup: {str(e)}")

    def _set_prediction_scheduler(self):
        """Set the prediction scheduler"""
        # Start the scheduler
        self.scheduler.start()

        # Configure a trigger for the scheduler (15 mins after the market close on weekdays)
        eastern = timezone("US/Eastern")
        trigger = CronTrigger(
            day_of_week="mon-fri", hour=16, minute=45, timezone=eastern
        )

        # Add job to the scheduler
        self.scheduler.add_job(self._predict_all, trigger=trigger)

    async def _predict_all(self):
        """Run batch prediction for all configured model types and symbols."""

        # Retrieve the model types
        trainers = await self.training_service.get_trainers()
        model_types = trainers["types"]

        # Retrieve the symbols to predict
        stocks_data = await self.data_service.get_nasdaq_stocks()
        symbols = [item["symbol"] for item in stocks_data["data"]]

        run_batch_prediction(
            model_types,
            symbols,
            self.data_service,
            self.data_processing_service,
            self.deployment_service,
        )
