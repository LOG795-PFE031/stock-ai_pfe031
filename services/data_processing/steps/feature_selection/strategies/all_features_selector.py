from .base_strategy import FeatureSelectionStrategy


class AllFeatureSelector(FeatureSelectionStrategy):
    """
    Strategy that returns all the features of the data frame (except the date).
    """

    def select(self, data):
        return data.drop(columns=["Date"])
