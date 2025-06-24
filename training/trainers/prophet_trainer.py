"""
Prophet model trainer.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

from core.logging import logger
from training.schemas import Metrics
from training.trainers.base_trainer import BaseTrainer
from training.trainer_registry import TrainerRegistry

# Constant for the trainer name
TRAINER_NAME = "prophet"


@TrainerRegistry.register(TRAINER_NAME)
class ProphetTrainer(BaseTrainer):
    """Trainer for Prophet models."""

    def __init__(self):
        super().__init__(TRAINER_NAME)
        self.logger = logger["training"]

    async def prepare_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data for Prophet model."""
        try:
            # Load stock data
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            df = pd.read_csv(data_file)

            # Convert date column to datetime and convert to UTC
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is not None:
                df["Date"] = df["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_localize(None)

            # Filter date range if specified
            if start_date:
                if start_date.tzinfo is not None:
                    start_date = start_date.astimezone(timezone.utc).replace(
                        tzinfo=None
                    )
                df = df[df["Date"] >= start_date]
            if end_date:
                if end_date.tzinfo is not None:
                    end_date = end_date.astimezone(timezone.utc).replace(tzinfo=None)
                df = df[df["Date"] <= end_date]

            # Prepare data for Prophet
            prophet_df = pd.DataFrame({"ds": df["Date"], "y": df["Close"]})

            # Add additional regressors
            for feature in self.config.model.FEATURES:
                if feature not in ["Close", "Date"]:
                    prophet_df[feature] = df[feature]

            # Split into train and test
            train_size = int(len(prophet_df) * 0.8)
            train_data = prophet_df[:train_size]
            test_data = prophet_df[train_size:]

            return train_data, test_data

        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise

    async def train(
        self, symbol: str, data: pd.DataFrame, **kwargs
    ) -> Tuple[Prophet, Dict[str, Any]]:
        """Train Prophet model."""
        try:
            # Initialize Prophet model
            model = Prophet(
                changepoint_prior_scale=kwargs.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=kwargs.get("seasonality_prior_scale", 10.0),
                holidays_prior_scale=kwargs.get("holidays_prior_scale", 10.0),
                seasonality_mode=kwargs.get("seasonality_mode", "multiplicative"),
            )

            # Add additional regressors
            for feature in self.config.model.FEATURES:
                if feature not in ["Close", "Date"]:
                    model.add_regressor(feature)

            # Fit model
            model.fit(data)

            # Get training history
            history = {
                "changepoints": (
                    model.changepoints.tolist()
                    if hasattr(model, "changepoints")
                    else []
                ),
                "trend": model.params["k"].tolist() if "k" in model.params else [],
                "seasonality": (
                    model.params["beta"].tolist() if "beta" in model.params else []
                ),
            }

            return model, history

        except Exception as e:
            self.logger.error(f"Error training Prophet model for {symbol}: {str(e)}")
            raise

    async def evaluate(self, model: Prophet, test_data: pd.DataFrame) -> Metrics:
        """Evaluate Prophet model."""
        try:
            # Make predictions
            forecast = model.predict(test_data)

            # Calculate metrics
            y_true = test_data["y"].values
            y_pred = forecast["yhat"].values

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            return Metrics(
                mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2)
            )

        except Exception as e:
            self.logger.error(f"Error evaluating Prophet model: {str(e)}")
            raise

    async def save_model(
        self, model: Prophet, symbol: str, metrics: Dict[str, float]
    ) -> None:
        """Save Prophet model."""
        try:
            # Create prophet directory if it doesn't exist
            prophet_dir = self.config.model.PROPHET_MODELS_DIR
            prophet_dir.mkdir(parents=True, exist_ok=True)

            # Save the actual Prophet model using joblib
            model_file = prophet_dir / f"{symbol}_prophet.joblib"
            joblib.dump(model, model_file)

            # Save model metadata
            model_data = {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_version,
            }

            # Save metadata as JSON
            metadata_path = prophet_dir / f"{symbol}_prophet_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(model_data, f, indent=4)

            self.logger.info(f"Saved Prophet model for {symbol}")

        except Exception as e:
            self.logger.error(f"Error saving Prophet model for {symbol}: {str(e)}")
            raise

    async def load_model(self, symbol: str) -> Optional[Prophet]:
        """Load Prophet model."""
        try:
            prophet_dir = self.model_dir / TRAINER_NAME
            model_file = prophet_dir / f"{symbol}_prophet.joblib"
            metadata_file = prophet_dir / f"{symbol}_prophet_metadata.json"

            if not model_file.exists():
                self.logger.warning(f"No Prophet model found for {symbol}")
                return None

            # Load the actual Prophet model
            model = joblib.load(model_file)

            # Load model metadata
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    model_data = json.load(f)
                    self.logger.info(f"Loaded Prophet model metadata for {symbol}")
                    self.logger.info(f"Model metrics: {model_data.get('metrics', {})}")

            return model

        except Exception as e:
            self.logger.error(f"Error loading Prophet model for {symbol}: {str(e)}")
            return None
