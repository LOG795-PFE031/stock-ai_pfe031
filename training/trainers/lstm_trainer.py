"""
LSTM model trainer.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from keras import Sequential, layers, callbacks  # Replace tensorflow import
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt

from core.logging import logger
from training.schemas import Metrics
from training.trainers.base_trainer import BaseTrainer
from training.trainer_registry import TrainerRegistry

# Constant for the trainer name
TRAINER_NAME = "lstm"


@TrainerRegistry.register(TRAINER_NAME)
class LSTMTrainer(BaseTrainer):
    """Trainer for LSTM models."""

    def __init__(self):
        super().__init__(TRAINER_NAME)
        self.scaler = MinMaxScaler()
        self.logger = logger["training"]

    async def prepare_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for LSTM model."""
        try:
            # Load stock data
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            df = pd.read_csv(data_file)

            # Convert date column
            df["Date"] = pd.to_datetime(df["Date"])

            # Filter date range if specified
            if start_date:
                df = df[df["Date"] >= start_date]
            if end_date:
                df = df[df["Date"] <= end_date]

            # Prepare features
            features_df = self._prepare_features(df)

            # Scale features
            scaled_data = self.scaler.fit_transform(features_df)

            # Create sequences
            X, y = self._create_sequences(
                scaled_data, self.config.model.SEQUENCE_LENGTH
            )

            # Split into train and test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            self.logger.info(f"Successfully prepared data for {symbol}")
            return (X_train, y_train), (X_test, y_test)

        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise

    async def train(
        self, symbol: str, data: Tuple[np.ndarray, np.ndarray], **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train LSTM model."""
        try:
            X_train, y_train = data

            model = self._build_model(X_train.shape[1:])

            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            history = model.fit(
                X_train,
                y_train,
                epochs=kwargs.get("epochs", 50),
                batch_size=kwargs.get("batch_size", 32),
                validation_split=0.2,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )
                ],
            )

            return model, history.history

        except Exception as e:
            self.logger.error(f"Error training LSTM model for {symbol}: {str(e)}")
            raise

    async def evaluate(
        self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Metrics:
        """Evaluate LSTM model."""
        try:
            X_test, y_test = test_data

            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            y_test = y_test[:, 2]  # Close price is the third column
            y_pred = y_pred.ravel()

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            return Metrics(
                mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2)
            )

        except Exception as e:
            self.logger.error(f"Error evaluating LSTM model: {str(e)}")
            raise

    async def save_model(
        self, model: Any, symbol: str, metrics: Dict[str, float]
    ) -> None:
        """Save LSTM model and scaler."""
        try:
            symbol_dir = self.model_dir / "specific" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            model_path = symbol_dir / f"{symbol}_model.keras"
            model.save(str(model_path))

            scaler_path = symbol_dir / f"{symbol}_scaler.gz"
            joblib.dump(self.scaler, scaler_path)

            scaler_metadata = {
                "feature_order": list(self.config.model.FEATURES),
                "min_values": self.scaler.data_min_.tolist(),
                "max_values": self.scaler.data_max_.tolist(),
                "feature_names": list(self.config.model.FEATURES),
                "sample_original": {
                    feature: float(self.scaler.data_min_[i])
                    for i, feature in enumerate(self.config.model.FEATURES)
                },
                "sample_scaled": {
                    feature: 0.0 for feature in self.config.model.FEATURES
                },
            }
            scaler_metadata_path = symbol_dir / f"{symbol}_scaler_metadata.json"
            with open(scaler_metadata_path, "w") as f:
                json.dump(scaler_metadata, f, indent=4)

            metrics["timestamp"] = datetime.now().isoformat()
            metrics["model_version"] = self.model_version
            metrics_path = symbol_dir / f"{symbol}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            # Save training history plot if available
            if hasattr(self, "history") and self.history is not None:
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 2, 1)
                plt.plot(self.history.history["loss"], label="Training Loss")
                plt.plot(self.history.history["val_loss"], label="Validation Loss")
                plt.title(f"{symbol} Model Training History")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(self.history.history["mae"], label="Training MAE")
                plt.plot(self.history.history["val_mae"], label="Validation MAE")
                plt.title(f"{symbol} Model Metrics")
                plt.xlabel("Epoch")
                plt.ylabel("MAE")
                plt.legend()

                plt.tight_layout()
                history_plot_path = symbol_dir / f"{symbol}_training_history.png"
                plt.savefig(history_plot_path)
                plt.close()

            self.logger.info(f"Saved LSTM model and metadata for {symbol}")

        except Exception as e:
            self.logger.error(f"Error saving LSTM model for {symbol}: {str(e)}")
            raise

    def _build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build LSTM model architecture."""
        model = Sequential(
            [
                layers.LSTM(
                    100,
                    return_sequences=True,
                    input_shape=input_shape,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                ),
                layers.Dropout(0.2),
                layers.LSTM(
                    50,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                ),
                layers.Dropout(0.2),
                layers.Dense(
                    50, activation="relu", kernel_initializer="glorot_uniform"
                ),
                layers.Dense(
                    25, activation="relu", kernel_initializer="glorot_uniform"
                ),
                layers.Dense(1),
            ]
        )

        return model

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save data as JSON."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
