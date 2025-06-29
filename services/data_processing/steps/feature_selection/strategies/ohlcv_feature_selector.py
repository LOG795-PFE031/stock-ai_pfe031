from .base_strategy import FeatureSelectionStrategy


class OHLCVFeatureSelector(FeatureSelectionStrategy):
    """
    Selector strategy that returns the OHLCV (Open, High, Low, Close, Volume)
    features of the data frame.
    """

    def select(self, data):
        features = ["Open", "High", "Low", "Close", "Volume"]
        return data[features]
