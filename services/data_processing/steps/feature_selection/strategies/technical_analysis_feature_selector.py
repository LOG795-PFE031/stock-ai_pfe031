from .base_strategy import FeatureSelectionStrategy


class TechnicalAnalysisFeatureSelector(FeatureSelectionStrategy):
    """
    Selector strategy that returns the Core Technical Analysis features
    (Open, High, Low, Close, Volume, Returns, MA_5, MA_20, RSI, MACD, MACD_Signal, Volatility).
    """

    def select(self, data):
        features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "MA_5",
            "MA_20",
            "Volatility",
            "RSI",
            "MACD",
            "MACD_Signal",
        ]
        return data[features]
