from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd

from core.logging import logger
from services import BaseService
from services import DataService
from .steps import (
    DataCleaner,
    DataNormalizer,
    DataSplitter,
    FeatureBuilder,
    FeatureSelector,
    InputFormatter,
)
from .scaler_manager import ScalerManager

from core.types import FormattedInput


class PreprocessingService(BaseService):

    def __init__(self, data_service: DataService):
        super().__init__()
        self.data_service = data_service
        self.logger = logger["preprocessing"]

    async def collect_preprocessed_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        pass

    async def fetch_scaler_path(
        self,
        symbol: str,
        model_type: str,
        phase: str,
    ) -> Path:
        """
        Fetch the path to the saved scaler for a given symbol, model type, and phase.

        Returns:
            Path: Path to the saved scaler file.
        """

        scaler_path = ScalerManager(
            model_type=model_type, symbol=symbol, phase=phase
        ).get_scaler_path()

        if scaler_path.exists():
            return scaler_path
        else:
            self.logger.error(f"Scaler file not found at path: {scaler_path}")
            raise FileNotFoundError()

    async def get_preprocessed_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_type: str,
        phase: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Union[FormattedInput, Tuple[FormattedInput, FormattedInput]]:
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

        # Clean the data
        clean_data = DataCleaner(symbol, self.logger).process(data)

        # Build features
        features = FeatureBuilder(symbol, self.logger).process(clean_data)

        # Select features
        features = FeatureSelector(symbol, self.logger, model_type).process(features)

        # Split the data
        if phase == "training":
            train_features, test_features = DataSplitter(symbol, self.logger).process(
                features
            )

            # Normalize the training data
            train_norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(train_features, fit=True)

            # Normalize the test data
            test_norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(test_features, fit=False)

            # Format the data to create the training input
            training_dataset = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(train_norm_features)

            # Format the data to create the test input
            test_dataset = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(test_norm_features)

            return training_dataset, test_dataset

        elif phase == "prediction":

            # Normalize the data
            norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(features)

            # Format the data to create the prediction input
            prediction_input = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(norm_features)

            return prediction_input

        # TODO store preproccessed Data (MinIO + Redis ?)
        # Save processed data
        """data_file = self.config.data.STOCK_DATA_DIR / f"processed_{symbol}.csv"
        df_processed.to_csv(data_file, index=False)
        return df_processed
        """
