"""
Base model class for model training.
"""

from abc import ABC
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel

from core.config import config
from core.logging import logger
from core.types import ProcessedData
from ..trainers import BaseTrainer
from .saving_strategies import BaseSaver


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

    async def train_and_save(self, data: ProcessedData) -> str:
        """
        Trains the model, saves it locally, and logs it to MLflow.

        Args:
            data (FormattedInput): Preprocessed training data.

        Returns:
            str: Name of the registered MLflow model.
        """

        try:
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

                # Register the model to MLFlow
                mlflow.pyfunc.log_model(
                    python_model=self.predictor,
                    artifact_path="model",
                    input_example=data.X,  # TODO To long to log. We will need to get one sample
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
