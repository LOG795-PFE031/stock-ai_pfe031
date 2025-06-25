from datetime import datetime
from typing import Optional

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

from .types import FormattedInput


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

    async def get_preprocessed_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_type: str,
        phase: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> FormattedInput:
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

            train_norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(train_features, fit=True)

            test_norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(test_features, fit=False)

            training_dataset = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(train_norm_features)

            test_dataset = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(test_norm_features)

            return training_dataset, test_dataset

        elif phase == "prediction":
            norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(features)

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
