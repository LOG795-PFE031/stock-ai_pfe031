from .base_strategy import FeatureSelectionStrategy


class SeasonalFeatureSelector(FeatureSelectionStrategy):
    """
    Selector strategy that returns features corresponding to
    seasonal analysis along with a set of technical indicators.
    """

    def select(self, data):
        features = [
            "Close",
            "Day_of_week",
            "Month",
            "Quarter",
            "MA_5",
            "MA_20",
            "MACD",
            "RSI",
        ]
        return data[features]
