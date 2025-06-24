from datetime import datetime
from typing import Optional

import pandas as pd

from core.logging import logger
from services import BaseService
from services import DataService


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

        # TODO Clean the data
        # TODO Build features
        # TODO etc..

        # TODO store preproccessed Data (MinIO + Redis ?)
        # Save processed data
        """data_file = self.config.data.STOCK_DATA_DIR / f"processed_{symbol}.csv"
        df_processed.to_csv(data_file, index=False)
        return df_processed
        """

        return
