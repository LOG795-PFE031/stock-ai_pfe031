from .base_strategy import FeatureSelectionStrategy


class ProphetFeatureSelector(FeatureSelectionStrategy):
    """
    Selector strategy for meta Prophet model
    """

    def select(self, data):

        # Rename the features Date and Close (to match Prophet Input)
        data.rename(columns={"Date": "ds"}, inplace=True)
        data["ds"] = data["ds"].dt.strftime("%Y-%m-%d")
        data.rename(columns={"Close": "y"}, inplace=True)

        features = [
            "y",
            "ds",
            "Open",
            "High",
            "Low",
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
