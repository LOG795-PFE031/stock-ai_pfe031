from keras import Sequential, layers, callbacks  # Replace tensorflow import
from typing import Any, Tuple, Dict
import numpy as np

from core.types import ProcessedData
from .base_trainer import BaseTrainer


class LSTMTrainer(BaseTrainer):
    """Trainer for LSTM models."""

    async def train(
        self, data: ProcessedData[np.ndarray], **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = data.X, data.y

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
            raise RuntimeError(
                f"Error occurred while training the LSTM model: {str(e)}"
            ) from e

    def _build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build LSTM model architecture."""
        model = Sequential(
            [
                layers.LSTM(
                    32,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    return_sequences=False,
                ),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )

        return model
