from services.data_processing.abstract import BaseDataProcessor
from .strategies import (
    AllFeatureSelector,
    OHLCVFeatureSelector,
    TechnicalAnalysisFeatureSelector,
    SeasonalFeatureSelector,
    ProphetFeatureSelector,
)
from .strategies.base_strategy import FeatureSelectionStrategy

import pandas as pd


class FeatureSelector(BaseDataProcessor):
    """
    Class that selects features for different model types using
    appropriate feature selection strategies.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_selector = self._get_feature_selector()
            data_selected = feature_selector.select(data)
            return data_selected
        except Exception as e:
            raise RuntimeError(
                f"Error selecting features for {self.model_type} model"
            ) from e

    def _get_feature_selector(self) -> FeatureSelectionStrategy:
        """
        Returns the appropriate feature selection strategy based on the model type.
        """

        if self.model_type == "lstm":
            return TechnicalAnalysisFeatureSelector()
        elif self.model_type == "prophet":
            return ProphetFeatureSelector()
        elif self.model_type == "xgboost":
            return AllFeatureSelector()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
