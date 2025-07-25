from keras import Sequential, layers, callbacks, Input  # Replace tensorflow import
from typing import Any, Tuple, Dict
import numpy as np

from core.types import LSTMInput
from .base_trainer import BaseTrainer


class LSTMTrainer(BaseTrainer):
    """Trainer for LSTM models."""

    async def train(self, data: LSTMInput, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        try:
            X_train, y_train = np.array(data.X), np.array(data.y)

            model = self._build_model(X_train.shape[1:])

            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            history = model.fit(
                X_train,
                y_train,
                shuffle=False,
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
                Input(input_shape),
                layers.LSTM(
                    32,
                    return_sequences=False,
                ),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )
        return model
