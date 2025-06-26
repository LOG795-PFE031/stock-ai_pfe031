from .scaler_factory import ScalerFactory
from services.preprocessing.scaler_manager import ScalerManager
from services.preprocessing.abstract import BaseDataProcessor

import numpy as np
import pandas as pd


class DataNormalizer(BaseDataProcessor):
    """
    Class that normalizes data for different model types using
    model-specific scalers during training and prediction.
    """

    def __init__(self, symbol, logger, model_type: str, phase: str):
        super().__init__(symbol, logger)
        self.model_type = model_type
        self.scaler_manager = ScalerManager(model_type, symbol, phase)

    def process(self, data: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Process the data by applying normalization if required.

        Args:
            data (pd.DataFrame): Input stock data.
            fit (bool): If True, fit a new scaler.

        Returns:
            pd.DataFrame: Processed stock data.
        """
        # Get (or create) the scaler
        scaler = self._load_or_fit_scaler(data, fit)

        if scaler:
            # Scale the data
            scaled = scaler.transform(data)
            return pd.DataFrame(scaled, columns=data.columns, index=data.index)
        else:
            # No Normalization needed
            return data

    def _load_or_fit_scaler(self, data, fit):
        """
        Returns the appropriate scaler based on the model type.
        """
        try:

            # Create it
            scaler = ScalerFactory.create_scaler(self.model_type)

            if scaler is None:
                self.logger.info(
                    f"No scaler required for model type '{self.model_type}'."
                )
                return None

            if fit:
                # Fit and save the scaler
                scaler.fit(data)
                self.scaler_manager.save_scaler(scaler)
            else:
                scaler = self.scaler_manager.load_scaler()

            return scaler

        except Exception as e:
            self.logger.error(
                f"Error preparing the scaler for data of symbol {self.symbol}: {e}"
            )
            raise e
