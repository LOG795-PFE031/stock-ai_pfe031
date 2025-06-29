from .base_strategy import FeatureSelectionStrategy


class AllFeatureSelector(FeatureSelectionStrategy):
    """
    Strategy that returns all the features of the data frame.
    """

    def select(self, data):
        return data
