"""
Base trainer class for model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# import mlflow
import json
import time

from core.config import config
from core.logging import logger
from core.utils import calculate_technical_indicators
from monitoring.prometheus_metrics import (
    data_points_ingested_total,
    evaluation_time_seconds,
    evaluation_mae,
    evaluation_mse,
    evaluation_r2,
    evaluation_rmse,
    model_saving_time_seconds,
    preprocessing_time_seconds,
    training_time_seconds,
)
from training.schemas import Metrics


class BaseTrainer(ABC):
    """Base class for all model trainers."""

    def __init__(self, model_type: str):
        self.config = config
        self.logger = logger
        self.model_type = model_type
        self.model_version = "0.1.0"
        self.model_dir = self.config.model.PREDICTION_MODELS_DIR

    @abstractmethod
    async def prepare_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data for the model."""
        pass

    @abstractmethod
    async def train(self, symbol: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    async def evaluate(self, model: Any, test_data: pd.DataFrame) -> Metrics:
        """Evaluate the model."""
        pass

    @abstractmethod
    async def save_model(
        self, model: Any, symbol: str, metrics: Dict[str, float]
    ) -> None:
        """Save the trained model."""
        pass

    async def train_and_evaluate(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train and evaluate the model.

        Args:
            symbol: Stock symbol
            start_date: Start date for training data
            end_date: End date for training data
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        try:
            start_time = time.perf_counter()  # Start timer

            # Prepare data
            train_data, test_data = await self.prepare_data(
                symbol, start_date, end_date
            )

            # Log the preprocessing time (Prometheus)
            data_points_ingested_total.labels(
                model_type=self.model_type, symbol=symbol
            ).inc(len(train_data[0]) + len(test_data[0]))

            # Log the preprocessing time (Prometheus)
            preprocessing_duration = time.perf_counter() - start_time
            preprocessing_time_seconds.labels(
                model_type=self.model_type, symbol=symbol
            ).observe(preprocessing_duration)

            start_time = time.perf_counter()  # Start timer

            # Train model
            model, training_history = await self.train(symbol, train_data, **kwargs)

            # Log the training time (Prometheus)
            training_duration = time.perf_counter() - start_time
            training_time_seconds.labels(
                model_type=self.model_type, symbol=symbol
            ).observe(training_duration)

            start_time = time.perf_counter()  # Start timer

            # Evaluate model
            metrics = await self.evaluate(model, test_data)

            # Log the evaluation metrics (Prometheus)
            evaluation_mae.labels(model_type=self.model_type, symbol=symbol).set(
                metrics.mae
            )
            evaluation_mse.labels(model_type=self.model_type, symbol=symbol).set(
                metrics.mse
            )
            evaluation_r2.labels(model_type=self.model_type, symbol=symbol).set(
                metrics.r2
            )
            evaluation_rmse.labels(model_type=self.model_type, symbol=symbol).set(
                metrics.rmse
            )

            # Format the metrics to a dict
            metrics = metrics.model_dump()

            # Log the evaluation time (Prometheus)
            evaluation_duration = time.perf_counter() - start_time
            evaluation_time_seconds.labels(
                model_type=self.model_type, symbol=symbol
            ).observe(evaluation_duration)

            start_time = time.perf_counter()  # Start timer

            # Save model
            await self.save_model(model, symbol, metrics)

            # Log the evaluation time (Prometheus)
            model_saving_duration = time.perf_counter() - start_time
            model_saving_time_seconds.labels(
                model_type=self.model_type, symbol=symbol
            ).observe(model_saving_duration)

            return {
                "symbol": symbol,
                "model_type": self.model_type,
                "model_version": self.model_version,
                "training_history": training_history,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {str(e)}")
            raise

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Select features
        features = self.config.model.FEATURES
        return df[features]

    def _create_sequences(
        self, data: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : (i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    def _save_metrics(self, symbol: str, metrics: Dict[str, float]) -> None:
        """Save training metrics."""
        metrics_file = self.model_dir / f"{symbol}_metrics.json"
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["model_version"] = self.model_version
        self.save_json(metrics, metrics_file)

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save data as JSON."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
