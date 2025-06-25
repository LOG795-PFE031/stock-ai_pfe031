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
    ) -> pd.DataFrame:
        """
        Preprocess the stock data for models input.

        Args:
            data (pd.DataFrame): Raw stock data

        Returns:
            pd.DataFrame: Processed Dataframe
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

            X_train, y_train = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(train_norm_features)

            X_test, y_test = InputFormatter(
                symbol, self.logger, model_type, phase
            ).process(test_norm_features)

            return X_train, y_train, X_test, y_test

        elif phase == "prediction":
            norm_features = DataNormalizer(
                symbol, self.logger, model_type, phase
            ).process(features)

            X = InputFormatter(symbol, self.logger, model_type, phase).process(
                norm_features
            )

            return X

        # TODO store preproccessed Data (MinIO + Redis ?)
        # Save processed data
        """data_file = self.config.data.STOCK_DATA_DIR / f"processed_{symbol}.csv"
        df_processed.to_csv(data_file, index=False)
        return df_processed
        """
