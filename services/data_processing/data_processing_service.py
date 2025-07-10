from datetime import datetime
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

from core.types import ProcessedData


class DataProcessingService(BaseService):

    def __init__(self):
        super().__init__()
        self.logger = logger["data_processing"]

    async def initialize(self) -> None:
        """Initialize the data processing service."""
        try:
            self._initialized = True

            # Map the phase to the correction function
            self.phase_function_map = {
                "training": self._preprocess_training_phase,
                "prediction": self._preprocess_prediction_phase,
                "evaluation": self._preprocess_evaluation_phase,
            }

            self.logger.info("Data processing service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data processing service: {str(e)}")
            raise

    async def promote_scaler(self, symbol: str, model_type: str):
        """
        Promote the training scalers to the prediction phase for a specific model and symbol.

        This copies both the features and targets scalers from the 'training' directory
        to the 'prediction' directory, if the model requires scaling.

        Args:
            symbol (str): The stock symbol
            model_type (str): The type of model
        """
        try:
            # Promote the scaler
            scaler_manager = ScalerManager(model_type=model_type, symbol=symbol)
            promoted = scaler_manager.promote_scaler()

            if promoted:
                self.logger.info(
                    f"Successfully promoted scalers for model '{model_type}' and symbol '{symbol}'."
                )
            else:
                self.logger.info(
                    f"No scaler promotion needed for model '{model_type}' and symbol '{symbol}'."
                )
        except Exception as e:
            self.logger.error(
                f"Failed to promote the scalers for {model_type} model for {symbol} : {str(e)}"
            )
            raise

    async def preprocess_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_type: str,
        phase: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Union[ProcessedData, Tuple[ProcessedData, ProcessedData]]:
        """
        Preprocess the stock data for models input.

        This method first applies common preprocessing steps that are independent of the phase.
        Then, it delegates to the appropriate phase-specific preprocessing function.

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

            if phase not in self.phase_function_map:
                raise ValueError(f"Unknown phase '{phase}'")

            # Clean the data
            clean_data = DataCleaner().process(data)
            self.logger.debug(
                f"Data cleaned for symbol={symbol}, model_type={model_type}"
            )

            # Retrieve dates
            dates = clean_data["Date"].dt.date

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

            # Capture the column-to-index map right after feature selection
            feature_index_map = {col: idx for idx, col in enumerate(features.columns)}

            # Format the data
            input_data = InputFormatter(model_type, phase).process(features)
            self.logger.debug(
                f"Data formatted for symbol={symbol}, model_type={model_type}, phase={phase}"
            )

            # Add the column-to-index map to the input data
            input_data.feature_index_map = feature_index_map

            # Execute the correction function based on the phase
            return self.phase_function_map[phase](input_data, dates, symbol, model_type)

        except Exception as e:
            self.logger.error(
                f"Error preprocessing the stock data for model {model_type} for {symbol} during {phase} phase: {str(e)}"
            )
            raise

    async def postprocess_data(
        self,
        symbol: str,
        prediction: pd.DataFrame,
        model_type: str,
        phase: str,
    ) -> ProcessedData:
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

            data = ProcessedData(y=prediction)

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

    def _preprocess_training_phase(
        self, features: ProcessedData, dates: pd.Series, symbol: str, model_type: str
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

        # Retrieve training dataset start and end dates
        train_start_date = dates[0]
        train_end_date = dates[len(norm_train_data.X) - 1]

        # Retrieve test dataset start and end dates
        test_start_date = dates[len(norm_train_data.X)]
        test_end_date = dates[len(dates) - 1]

        # Add start and end dates to preprocessed data (Train and test)
        norm_train_data.start_date = train_start_date
        norm_train_data.end_date = train_end_date

        norm_test_data.start_date = test_start_date
        norm_test_data.end_date = test_end_date

        self.logger.info(
            f"Completed training phase preprocessing for {symbol} symbol for {model_type} model"
        )

        return norm_train_data, norm_test_data

    def _preprocess_evaluation_phase(
        self, features: ProcessedData, dates: pd.Series, symbol: str, model_type: str
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

        # Retrieve the start and end dates
        eval_start_date = dates[len(dates) - len(norm_eval_data.X)]
        eval_end_date = dates[len(dates) - 1]

        # Add start and end dates to preprocessed data
        norm_eval_data.start_date = eval_start_date
        norm_eval_data.end_date = eval_end_date

        self.logger.info(
            f"Completed evaluation phase preprocessing for {symbol} symbol for model {model_type}"
        )

        return norm_eval_data

    def _preprocess_prediction_phase(
        self, features: ProcessedData, dates: pd.Series, symbol: str, model_type: str
    ):
        """Handle the preprocessing steps for prediction."""

        # Normalize the data
        norm_features = DataNormalizer(symbol, model_type, "prediction").process(
            features, fit=False
        )

        self.logger.debug(
            f"Prediction data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Retrieve the start and end dates
        pred_start_date = dates[len(dates) - len(norm_features.X)]
        pred_end_date = dates[len(dates) - 1]

        # Add start and end dates to preprocessed data
        norm_features.start_date = pred_start_date
        norm_features.end_date = pred_end_date

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
