"""
Base model class for model training.
"""

from abc import ABC
from pathlib import Path
from typing import Union

import mlflow
from mlflow.pyfunc import PythonModel

from core.config import config
from core.logging import logger
from core.types import LSTMInput, ProphetInput, XGBoostInput
from ..trainers import BaseTrainer
from .saving_strategies import BaseSaver
from mlflow.pyfunc import PythonModel


class BaseModel(ABC, PythonModel):
    """Base class for all model trainers."""

    def __init__(
        self,
        model_type: str,
        symbol: str,
        saver: BaseSaver,
        trainer: BaseTrainer,
        predictor: PythonModel,
    ):
        self.model_type = model_type
        self.symbol = symbol

        self.saver = saver
        self.trainer = trainer
        self.predictor = predictor

        self.config = config
        self.logger = logger["training"]
        self.model_root_dir = self.config.model.MODELS_ROOT_DIR

    async def train_and_save(
        self, data: Union[LSTMInput, ProphetInput, XGBoostInput]
    ) -> str:
        """
        Trains the model, saves it locally, and logs it to MLflow.

        Args:
            data (FormattedInput): Preprocessed training data.

        Returns:
            str: Name of the registered MLflow model.
        """

        try:

            class LSTMPredictor(PythonModel):
                def load_context(self, context):
                    from keras import models  # Replace tensorflow import

                    model_path = context.artifacts.get("model")
                    if not model_path:
                        raise ValueError(
                            "Model path for LSTM model is missing from MLflow artifacts."
                        )

                    self.model = models.load_model(model_path, compile=False)

                def predict(self, context, model_input, params=None):
                    return self.model.predict(model_input)

            class ProphetPredictor(PythonModel):
                def load_context(self, context):
                    import joblib

                    model_path = context.artifacts.get("model")
                    if not model_path:
                        raise ValueError(
                            "Model path for the Prophet model is missing from MLflow artifacts."
                        )
                    self.model = joblib.load(model_path)

                def predict(self, context, model_input, params=None):
                    return self.model.predict(model_input)

            class XGBoostPredictor(PythonModel):
                def load_context(self, context):
                    import joblib

                    model_path = context.artifacts.get("model")
                    if not model_path:
                        raise ValueError(
                            "Model path for the XGBoost model is missing from MLflow artifacts."
                        )
                    self.model = joblib.load(model_path)

                def predict(self, context, model_input, params=None):
                    return self.model.predict(model_input)

            model_map = {
                "lstm": LSTMPredictor,
                "xgboost": XGBoostPredictor,
                "prophet": ProphetPredictor,
            }

            mlflow.set_experiment("training_experiments")

            with mlflow.start_run() as run:

                # Set custom tags
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("model_type", self.model_type)
                mlflow.set_tag("symbol", self.symbol)
                mlflow.set_tag("training_data_start_date", data.start_date)
                mlflow.set_tag("training_data_end_date", data.end_date)

                # Train the model
                model, training_history = await self.trainer.train(data)

                # Save the model (locally)
                saved_training_model_path = await self.saver.save(
                    model, base_path=self._get_training_model_dir()
                )

                python_model = model_map.get(self.model_type)
                # Register the model to MLFlow
                mlflow.pyfunc.log_model(
                    python_model=python_model(),
                    artifact_path="model",
                    artifacts={"model": str(saved_training_model_path)},
                )

            return {
                "run_id": run.info.run_id,
                "run_info": run.info.__dict__,
                "training_history": training_history,
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to train and save model '{self.model_type}': {e}"
            ) from e

    def _get_training_model_dir(self) -> Path:
        """Return the path to the model training directory"""
        return self.model_root_dir / self.model_type / self.symbol
