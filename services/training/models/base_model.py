"""
Base model class for model training.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel

from .saving_strategies import BaseSaver
from core.config import config
from core.types import PreprocessedData
from core.utils import get_model_name
from core.logging import logger
from services.training.trainers import BaseTrainer


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
        self.model_name = get_model_name(model_type, symbol, "training")

    async def train_and_save(self, data: PreprocessedData) -> str:
        """
        Trains the model, saves it locally, and logs it to MLflow.

        Args:
            data (FormattedInput): Preprocessed training data.

        Returns:
            str: Name of the registered MLflow model.
        """

        try:
            with mlflow.start_run() as run:

                # Set custom tags
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("model_type", self.model_type)
                mlflow.set_tag("symbol", self.symbol)

                # Train the model
                model, training_history = await self.trainer.train(data)

                # Save the model (locally)
                saved_training_model_path = await self.saver.save(
                    model, base_path=self._get_training_model_dir()
                )

                # Register the model to MLFlow
                mlflow.pyfunc.log_model(
                    python_model=self.predictor,
                    artifact_path=self.model_name,
                    input_example=data.X,  # TODO To long to log. We will need to get one sample
                    registered_model_name=self.model_name,
                    artifacts={"model": str(saved_training_model_path)},
                )

            return {
                "model_name": self.model_name,
                "run_info": run.info,
                "training_history": training_history,
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to train and save model '{self.model_name}': {e}"
            ) from e

    def _get_training_model_dir(self) -> Path:
        """Return the path to the model training directory"""
        return self.model_root_dir / self.model_type / "training" / self.symbol
