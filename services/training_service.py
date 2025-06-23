"""
Training service for model training and evaluation.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from services.base_service import BaseService
from training.lstm_trainer import LSTMTrainer
from training.prophet_trainer import ProphetTrainer
from core.utils import validate_stock_symbol
from core.logging import logger
from monitoring.prometheus_metrics import training_total
from monitoring.utils import monitor_training_cpu_usage, monitor_training_memory_usage


class TrainingService(BaseService):
    """Service for model training and evaluation."""

    def __init__(self, model_service, data_service):
        super().__init__()
        self._initialized = False
        self.trainers = {"lstm": LSTMTrainer(), "prophet": ProphetTrainer()}
        self.training_tasks = {}
        self.scalers = {}
        self.model_version = "0.1.0"
        self.model_service = model_service
        self.data_service = data_service
        self.logger = logger["training"]

    async def initialize(self) -> None:
        """Initialize the training service."""
        try:
            self._initialized = True
            self.logger.info("Training service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize training service: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel any running training tasks
            for task in self.training_tasks.values():
                if not task.done():
                    task.cancel()

            self.training_tasks.clear()
            self._initialized = False
            self.logger.info("Training service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during training service cleanup: {str(e)}")

    async def train_model(
        self,
        symbol: str,
        model_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train a model for a given symbol.

        Args:
            symbol: Stock symbol
            model_type: Type of model to train (lstm or prophet)
            start_date: Start date for training data
            end_date: End date for training data
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        if not self._initialized:
            return {
                "status": "error",
                "error": "Training service not initialized",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if not validate_stock_symbol(symbol):
            return {
                "status": "error",
                "error": f"Invalid stock symbol: {symbol}",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if model_type not in self.trainers:
            return {
                "status": "error",
                "error": f"Unsupported model type: {model_type}",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Begin to monitor cpu usage
            monitor_task = asyncio.create_task(
                monitor_training_cpu_usage(model_type, symbol)
            )

            # Begin to monitor memory usage
            monitor_task = asyncio.create_task(
                monitor_training_memory_usage(model_type, symbol)
            )

            # Create training task
            task = asyncio.create_task(
                self.trainers[model_type].train_and_evaluate(
                    symbol, start_date, end_date, **kwargs
                )
            )

            # Store task
            task_key = f"{symbol}_{model_type}"
            self.training_tasks[task_key] = task

            # Wait for completion
            result = await task

            # Stop the monitor after training completes
            monitor_task.cancel()

            # Remove completed task
            del self.training_tasks[task_key]

            # Log the sucessful training count (Prometheus)
            training_total.labels(
                model_type=model_type, symbol=symbol, result="sucess"
            ).inc()

            return {
                "status": "success",
                "symbol": symbol,
                "model_type": model_type,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(
                f"Error training {model_type} model for {symbol}: {str(e)}"
            )

            # Log the unsucessful training count (Prometheus)
            training_total.labels(
                model_type=model_type, symbol=symbol, result="error"
            ).inc()

            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_training_status(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """
        Get training status for a model.

        Args:
            symbol: Stock symbol
            model_type: Type of model

        Returns:
            Dictionary containing training status
        """
        task_key = f"{symbol}_{model_type}"
        task = self.training_tasks.get(task_key)

        if not task:
            return {
                "status": "not_found",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if task.done():
            try:
                result = task.result()
                return {
                    "status": "completed",
                    "symbol": symbol,
                    "model_type": model_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "symbol": symbol,
                    "model_type": model_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
        else:
            return {
                "status": "in_progress",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def cancel_training(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """
        Cancel training for a model.

        Args:
            symbol: Stock symbol
            model_type: Type of model

        Returns:
            Dictionary containing cancellation status
        """
        task_key = f"{symbol}_{model_type}"
        task = self.training_tasks.get(task_key)

        if not task:
            return {
                "status": "not_found",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if not task.done():
            task.cancel()
            del self.training_tasks[task_key]
            return {
                "status": "cancelled",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "status": "already_completed",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_active_training_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of active training tasks.

        Returns:
            List of dictionaries containing task information
        """
        return [
            {
                "symbol": task_key.split("_")[0],
                "model_type": task_key.split("_")[1],
                "status": "in_progress" if not task.done() else "completed",
                "timestamp": datetime.now().isoformat(),
            }
            for task_key, task in self.training_tasks.items()
        ]
