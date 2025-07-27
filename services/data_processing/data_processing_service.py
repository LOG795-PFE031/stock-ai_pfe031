from typing import Union, Any, List, Dict
import pandas as pd
import numpy as np

from core.logging import logger
from .steps.formatters import ProcessedData
from core import BaseService
from .scaler_manager import ScalerManager
from .steps import (
    DataCleaner,
    DataNormalizer,
    DataSplitter,
    FeatureBuilder,
    FeatureSelector,
    InputFormatter,
    OutputFormatter,
)

from .schemas import SplitProcessedData, SingleProcessedData


class DataProcessingService(BaseService):
    """
    Service responsible for end-to-end preprocessing and postprocessing of stock data
    for different machine learning model types (e.g., LSTM, Prophet, XGBoost) across
    training, prediction, and evaluation phases.

    Responsibilities include:
    - Cleaning and transforming raw stock data into model-ready format.
    - Selecting and formatting features according to the model type and phase.
    - Normalizing and unnormalizing data using model-specific scalers.
    - Managing phase-specific logic (e.g., data splitting or promotion of scalers).

    """

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

    def promote_scaler(self, symbol: str, model_type: str) -> dict:
        """
        Promote the training scalers to the prediction phase for a specific model and symbol.

        This copies both the features and targets scalers from the 'training' directory
        to the 'prediction' directory, if the model requires scaling.

        Args:
            symbol (str): The stock symbol
            model_type (str): The type of model

        Returns:
            dict: Promotion status and message.
        """
        try:
            # Promote the scaler
            scaler_manager = ScalerManager(model_type=model_type, symbol=symbol)
            promoted = scaler_manager.promote_scaler()

            if promoted:
                message = f"Successfully promoted scalers for model {model_type} and symbol {symbol}."
                self.logger.info(message)
                return {"status": "success", "promoted": True, "message": message}
            else:
                message = f"No scaler promotion needed for model {model_type} and symbol {symbol}."
                self.logger.info(message)
                return {"status": "success", "promoted": False, "message": message}

        except Exception as e:
            error_msg = f"Failed to promote the scalers for {model_type} model for {symbol} : {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def preprocess_data(
        self,
        symbol: str,
        raw_data: List[Dict[str, Any]],
        model_type: str,
        phase: str,
    ) -> Union[SingleProcessedData, SplitProcessedData]:
        """
        Preprocess the stock data for models input (raw data to preprocessed data).

        This method first applies common preprocessing steps that are independent of the phase.
        Then, it delegates to the appropriate phase-specific preprocessing function.

        Args:
            symbol (str): Stock symbol
            raw_data (List[Dict[str, Any]]): Raw stock data
            model_type (str): Type of model
            phase (str): The phase (e.g., "training", "prediction").

        Returns:
            FormattedInput: Processed data formatted specifically for input into a model.
        """
        try:
            # Transform the raw data into a pandas DataFrame
            raw_data_df = pd.DataFrame(raw_data)

            self.logger.info(
                f"Starting preprocessing for symbol={symbol} for model {model_type} during {phase} phase"
            )

            if phase not in self.phase_function_map:
                raise ValueError(f"Unknown phase '{phase}'")

            # Clean the data
            clean_data = DataCleaner().process(raw_data_df)
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
        prediction,
        model_type: str,
        phase: str,
    ) -> ProcessedData:
        """
        Postprocess the predictions by reversing the scaling applied during preprocessing.

        Args:
            symbol (str): Stock symbol
            prediction (Any): Scaled predictions to be unnormalized.
            model_type (str): Type of model
            phase (str): The phase (e.g., "training", "prediction").

        Returns:
            FormattedInput: Processed data including the unscaled targets
        """
        try:

            self.logger.info(
                f"Starting postprocessing for {symbol} symbol for {model_type} model during {phase} phase"
            )

            # Format the predictions into either a numpy array or dataframe
            if isinstance(prediction, list):
                if isinstance(prediction[0], dict):
                    formatted_prediction = pd.DataFrame(prediction)
                else:
                    formatted_prediction = np.array(prediction)
            else:
                # No valid prediction given
                raise ValueError("Invalid prediction format (expect a list)")

            data = ProcessedData(y=formatted_prediction)

            data = DataNormalizer(symbol=symbol, model_type=model_type).unprocess(
                data, phase=phase
            )
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

            # Just return the targets (postprocessed predictions)
            return SingleProcessedData(
                X=None,
                y=self._format_input_data(data.y),
                feature_index_map=None,
                start_date=None,
                end_date=None,
            )
        except Exception as e:
            self.logger.error(
                f"Error postprocessing the stock data for {model_type} model for {symbol} during {phase} phase: {str(e)}"
            )
            raise

    def _preprocess_training_phase(
        self,
        features: ProcessedData,
        dates: pd.Series,
        symbol: str,
        model_type: str,
    ) -> SplitProcessedData:
        """Handle the preprocessing steps for training."""

        # Split the data
        train_data, test_data = DataSplitter().process(features)
        self.logger.debug(
            f"Data split completed for symbol={symbol}, model_type={model_type} — "
            f"train size: {len(train_data.X) if train_data.X is not None else 0}, "
            f"test size: {len(test_data.X) if test_data.X is not None else 0}"
        )

        # Retrieve training dataset start and end dates
        train_start_date = dates.iloc[0]
        train_end_date = dates.iloc[len(train_data.X) - 1]

        # Retrieve test dataset start and end dates
        test_start_date = dates.iloc[len(train_data.X)]
        test_end_date = dates.iloc[len(dates) - 1]

        # Normalize the training data
        norm_train_data = DataNormalizer(symbol=symbol, model_type=model_type).process(
            train_data, fit=True, scaler_dates=(train_start_date, train_end_date)
        )
        self.logger.debug(
            f"Training data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Normalize the test data
        norm_test_data = DataNormalizer(symbol, model_type).process(
            test_data, phase="training"
        )

        self.logger.debug(
            f"Test data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Add start and end dates to preprocessed data (Train and test)
        norm_train_data.start_date = train_start_date
        norm_train_data.end_date = train_end_date

        norm_test_data.start_date = test_start_date
        norm_test_data.end_date = test_end_date

        # Format the training dataset (preprocessed)
        training_dataset = SingleProcessedData(
            X=self._format_input_data(norm_train_data.X),
            y=self._format_input_data(norm_train_data.y),
            feature_index_map=norm_train_data.feature_index_map,
            start_date=norm_train_data.start_date.isoformat(),
            end_date=norm_train_data.end_date.isoformat(),
        )

        # Format the test dataset (preprocessed)
        test_dataset = SingleProcessedData(
            X=self._format_input_data(norm_test_data.X),
            y=self._format_input_data(norm_test_data.y),
            feature_index_map=norm_test_data.feature_index_map,
            start_date=norm_test_data.start_date.isoformat(),
            end_date=norm_test_data.end_date.isoformat(),
        )

        self.logger.info(
            f"Completed training phase preprocessing for {symbol} symbol for {model_type} model"
        )

        return SplitProcessedData(train=training_dataset, test=test_dataset)

    def _preprocess_evaluation_phase(
        self,
        features: ProcessedData,
        dates: pd.Series,
        symbol: str,
        model_type: str,
    ) -> SingleProcessedData:
        """Handle the preprocessing steps for evaluation"""

        # Split the data
        _, eval_data = DataSplitter().process(features)
        self.logger.debug(
            f"Data split completed for symbol={symbol}, model_type={model_type} — "
            f"Eval size: {len(eval_data.X) if eval_data.X is not None else 0}"
        )

        # Normalize the test data (use prediction-phase scaler)
        norm_eval_data = DataNormalizer(symbol, model_type).process(
            eval_data, phase="prediction"
        )

        self.logger.debug(
            f"Evaluation data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Retrieve the start and end dates
        eval_start_date = dates.iloc[len(dates) - len(norm_eval_data.X)]
        eval_end_date = dates.iloc[len(dates) - 1]

        # Format the preprocessed data
        eval_preprocessed = SingleProcessedData(
            X=self._format_input_data(norm_eval_data.X),
            y=self._format_input_data(norm_eval_data.y),
            feature_index_map=norm_eval_data.feature_index_map,
            start_date=eval_start_date.isoformat(),
            end_date=eval_end_date.isoformat(),
        )

        self.logger.info(
            f"Completed evaluation phase preprocessing for {symbol} symbol for model {model_type}"
        )

        return eval_preprocessed

    def _preprocess_prediction_phase(
        self,
        features: ProcessedData,
        dates: pd.Series,
        symbol: str,
        model_type: str,
    ) -> SingleProcessedData:
        """Handle the preprocessing steps for prediction."""

        # Normalize the data
        norm_features = DataNormalizer(symbol, model_type).process(
            features, phase="prediction"
        )

        self.logger.debug(
            f"Prediction data normalized for symbol={symbol}, model_type={model_type}"
        )

        # Retrieve the start and end dates
        pred_start_date = dates.iloc[len(dates) - len(norm_features.X)]
        pred_end_date = dates.iloc[len(dates) - 1]

        # Format the preprocessed data
        pred_preprocessed = SingleProcessedData(
            X=self._format_input_data(norm_features.X),
            y=self._format_input_data(norm_features.y),
            feature_index_map=norm_features.feature_index_map,
            start_date=pred_start_date.isoformat(),
            end_date=pred_end_date.isoformat(),
        )

        self.logger.info(
            f"Completed prediction phase preprocessing for {symbol} symbol for {model_type} model"
        )

        return pred_preprocessed

    def _format_input_data(self, input_data):
        """
        Format input features or predictions into a JSON-serializable structure.

        Args:
            input_data: A pandas DataFrame, NumPy array, or list of values.

        Returns:
            list or list of dict: A structure suitable for JSON serialization, typically used for API payloads.
                - If input is a pandas DataFrame, returns a list of dicts (records).
                - If input is a NumPy array, returns a nested list.
                - If input is already a list, returns it as-is.
        """
        if hasattr(input_data, "tolist"):
            return input_data.tolist()
        elif hasattr(input_data, "to_dict"):
            return input_data.to_dict(orient="records")
        elif isinstance(input_data, list):
            return input_data
        else:
            return None

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Preprocessing service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during preprocessing service cleanup: {str(e)}")
