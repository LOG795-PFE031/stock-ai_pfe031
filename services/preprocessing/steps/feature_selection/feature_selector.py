from services.preprocessing.abstract import BaseDataProcessor
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

    def __init__(self, symbol, logger, model_type: str):
        super().__init__(symbol, logger)
        self.model_type = model_type

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_selector = self._get_feature_selector()
        return feature_selector.select(data)

    def _get_feature_selector(self) -> FeatureSelectionStrategy:
        """
        Returns the appropriate feature selection strategy based on the model type.
        """

        if self.model_type == "lstm":
            return TechnicalAnalysisFeatureSelector()
        elif self.model_type == "prophet":
            return ProphetFeatureSelector()
        elif self.model_type == "xgboost":
            raise NotImplementedError(
                f"No feature selection stratagy implemented for the model {self.model_type}"
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
