"""
Training service for model training and evaluation.
"""

from typing import Dict, Any, List, Union
from datetime import datetime
import time
import asyncio
import os
import random
import numpy as np
import tensorflow as tf

from .model_registry import ModelRegistry
from core import BaseService
from core.utils import validate_stock_symbol
from core.types import LSTMInput, ProphetInput, XGBoostInput
from core.logging import logger
from . import models  # Dynamically imports models modules


class TrainingService(BaseService):
    """Service for model training and evaluation."""

    def __init__(self):
        super().__init__()
        self._initialized = False
        self.training_tasks = {}
        self.logger = logger["training"]

    async def initialize(self) -> None:
        """Initialize the training service."""
        try:
            self._enable_full_reproducibility()
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

    async def get_trainers(self) -> Dict[str, Any]:
        """Retrieve the list of available training trainers (from the TrainerFactory)"""
        try:
            self.logger.info("Starting to retrieve the list of available trainers.")
            trainers = ModelRegistry.list_models()
            self.logger.info(
                f"Successfully retrieved {len(trainers)} trainers from ModelRegistry."
            )
            return {
                "status": "success",
                "types": trainers,
                "count": len(trainers),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting the trainers : {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def train_model(
        self,
        symbol: str,
        model_type: str,
        data: Union[LSTMInput, ProphetInput, XGBoostInput],
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
            self.logger.error("Training service not initialized.")
            return {
                "status": "error",
                "error": "Training service not initialized",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if not validate_stock_symbol(symbol):
            self.logger.error(f"Invalid stock symbol: {symbol}")
            return {
                "status": "error",
                "error": f"Invalid stock symbol: {symbol}",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if model_type not in ModelRegistry.list_models():
            self.logger.error(f"Unsupported model type: {model_type}")
            return {
                "status": "error",
                "error": f"Unsupported model type: {model_type}",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        if not data or data.X is None:
            self.logger.error(f"No valid data for model training for {symbol}.")
            return {
                "status": "error",
                "error": f"No valid data for model training for {symbol}",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Create the model
            model = ModelRegistry.create(name=model_type, symbol=symbol)

            # Create training task
            task = asyncio.create_task(model.train_and_save(data))
            self.logger.debug(
                f"Training task for {model_type} model for {symbol} has started."
            )

            # Store task
            task_key = f"{symbol}_{model_type}"
            self.training_tasks[task_key] = task

            # Wait for training completion
            result = await task

            self.logger.info(f"Training completed for {model_type} model for {symbol}.")

            # Remove completed task
            del self.training_tasks[task_key]

            self.logger.info(
                f"Training successful for {model_type} model for {symbol}."
            )

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                **result,
            }

        except Exception as e:
            self.logger.error(
                f"Error training {model_type} model for {symbol}: {str(e)}"
            )

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

    def _enable_full_reproducibility(self, seed: int = 42):
        """
        Ensures reproducibility by setting random seeds and configuring TensorFlow.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
