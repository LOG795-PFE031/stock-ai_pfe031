from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd

from core.logging import logger
from services import BaseService
from .steps import (
    DataCleaner,
    DataNormalizer,
    DataSplitter,
    FeatureBuilder,
    FeatureSelector,
    InputFormatter,
    OutputFormatter,
)
from .scaler_manager import ScalerManager

from core.types import PreprocessedData


class DataProcessingService(BaseService):

    def __init__(self):
        super().__init__()
        self.logger = logger["data_processing"]

    async def initialize(self) -> None:
        """Initialize the data processing service."""
        try:
            self._initialized = True
            self.logger.info("Data processing service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data processing service: {str(e)}")
            raise

    async def fetch_scaler_path(
        self,
        symbol: str,
        model_type: str,
        phase: str,
        scaler_type=ScalerManager.FEATURES_SCALER_TYPE,
    ) -> Path:
        """
        Fetch the path to the saved scaler for a given symbol, model type, and phase.

        Returns:
            Path: Path to the saved scaler file.
        """

        scaler_path = ScalerManager(
            model_type=model_type, symbol=symbol, phase=phase
        ).get_scaler_path(scaler_type)

        if scaler_path.exists():
            return scaler_path
        else:
            self.logger.error(f"Scaler file not found at path: {scaler_path}")
            raise FileNotFoundError()

    async def preprocess_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_type: str,
        phase: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Union[PreprocessedData, Tuple[PreprocessedData, PreprocessedData]]:
        """
        Preprocess the stock data for models input.

        Args:
            data (pd.DataFrame): Raw stock data

        Returns:
            FormattedInput: Processed data formatted specifically for input into a model.
        """

        # TODO Check existing preproccessed Data (MinIO + Redis ? Or Cache ?)
        """
        Exemple:
        if preprocessing_is_cached(symbol, start_date, end_date)
            return preprocessed data
        else
            generate preprocessed data
        """

        try:

            self.logger.info(
                f"Starting preprocessing for symbol={symbol} for model {model_type} during {phase} phase"
            )

            # Clean the data
            clean_data = DataCleaner().process(data)
            self.logger.debug(
                f"Data cleaned for symbol={symbol}, model_type={model_type}"
            )

            # Build features
            features = FeatureBuilder().process(clean_data)
            self.logger.debug(
                f"Features built for symbol={symbol}, model_type={model_type}"
            )

            # Select features
            features = FeatureSelector(model_type).process(features)
            self.logger.debug(
                f"Features selected for symbol={symbol}, model_type={model_type}"
            )

            # Format the data
            features = InputFormatter(model_type, phase).process(features)
            self.logger.debug(
                f"Data formatted for symbol={symbol}, model_type={model_type}, phase={phase}"
            )

            if phase == "training":
                return await self._preprocess_training_phase(
                    features, symbol, model_type
                )

            elif phase == "prediction":
                return await self._preprocess_prediction_phase(
                    features, symbol, model_type
                )

            elif phase == "evaluation":
                return await self._preprocess_evaluation_phase(
                    features, symbol, model_type
                )

            else:
                raise ValueError(
                    f"Unknown phase '{phase}'. Expected one of: 'training', 'prediction', or 'evaluation'."
                )

        except Exception as e:
            self.logger.error(
                f"Error preprocessing the stock data for model {model_type} for {symbol} during {phase} phase: {str(e)}"
            )
            raise

    async def postprocess_data(
        self,
        symbol: str,
        targets: pd.DataFrame,
        model_type: str,
        phase: str,
    ) -> PreprocessedData:
        """
        Postprocess the predictions by reversing the scaling applied during preprocessing.

        Args:
            symbol (str): Stock symbol
            targets (pd.DataFrame): Scaled predictions to be unnormalized.
            model_type: Type of model
            phase (str): The phase (e.g., "train", "test", or "inference").

        Returns:
            FormattedInput: Processed data including the unscaled targets
        """
        try:
            self.logger.info(
                f"Starting postprocessing for {symbol} symbol for {model_type} model during {phase} phase"
            )

            data = PreprocessedData(y=targets)

            data = DataNormalizer(
                symbol=symbol, model_type=model_type, phase=phase
            ).unprocess(data)
            self.logger.debug(
                f"Targets unnormalized for symbol={symbol}, model_type={model_type}"
            )

            data = OutputFormatter(model_type=model_type).process(data)
            self.logger.debug(
                f"Targets formatted for symbol={symbol}, model_type={model_type}"
            )

            self.logger.info(
                f"Postprocessing completed for {symbol} symbol for {model_type} model during {phase} phase"
            )

            return data
        except Exception as e:
            self.logger.error(
                f"Error postprocessing the stock data for {model_type} model for {symbol} during {phase} phase: {str(e)}"
            )
            raise

    async def _preprocess_training_phase(
        self, features: PreprocessedData, symbol: str, model_type: str
    ):
        """Handle the preprocessing steps for training."""

        # Split the data
        train_data, test_data = DataSplitter().process(features)
        self.logger.debug(
            f"Data split completed for symbol={symbol}, model_type={model_type} — "
            f"train size: {len(train_data.X) if train_data.X is not None else 0}, "
            f"test size: {len(test_data.X) if test_data.X is not None else 0}"
        )

        # Normalize the training data
        norm_train_data = DataNormalizer(symbol, model_type, "training").process(
            train_data, fit=True
        )
        self.logger.debug(
            f"Training data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Normalize the test data
        norm_test_data = DataNormalizer(symbol, model_type, "training").process(
            test_data, fit=False
        )
        self.logger.debug(
            f"Test data normalized for symbol={symbol}, model_type={model_type}"
        )

        self.logger.info(
            f"Completed training phase preprocessing for {symbol} symbol for {model_type} model"
        )

        return norm_train_data, norm_test_data

    async def _preprocess_evaluation_phase(
        self, features: PreprocessedData, symbol: str, model_type: str
    ):
        """Handle the preprocessing steps for evaluation"""

        # Split the data
        _, eval_data = DataSplitter().process(features)
        self.logger.debug(
            f"Data split completed for symbol={symbol}, model_type={model_type} — "
            f"Eval size: {len(eval_data.X) if eval_data.X is not None else 0}"
        )

        # Normalize the test data (use prediction-phase scaler)
        norm_eval_data = DataNormalizer(symbol, model_type, "prediction").process(
            eval_data, fit=False
        )
        self.logger.debug(
            f"Evaluation data normalized for symbol={symbol}, model_type={model_type}"
        )

        self.logger.info(
            f"Completed evaluation phase preprocessing for {symbol} symbol for model {model_type}"
        )

        return norm_eval_data

    async def _preprocess_prediction_phase(
        self, features: PreprocessedData, symbol: str, model_type: str
    ):
        """Handle the preprocessing steps for prediction."""

        # Normalize the data
        norm_features = DataNormalizer(symbol, model_type, "prediction").process(
            features, fit=False
        )
        self.logger.debug(
            f"Prediction data normalized for symbol={symbol}, model_type={model_type}"
        )

        self.logger.info(
            f"Completed prediction phase preprocessing for {symbol} symbol for {model_type} model"
        )

        return norm_features

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Preprocessing service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during preprocessing service cleanup: {str(e)}")
