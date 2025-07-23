from .base_strategy import InputFormatterStrategy
from core.types import ProcessedData


class XGBoostInputFormatter(InputFormatterStrategy):
    def format(self, data, phase):
        if phase == "prediction":

            # Get the latest data
            latest_data = data.tail(1).reset_index(drop=True)
            return ProcessedData(X=latest_data)

        elif phase == "training" or "evaluation":
            # Add a column representing the target price (shift to get the price in the future)
            data["Target"] = data["Close"].shift(-1)

            # Remove rows containing na values (usually the most recent row (last one))
            data.dropna(inplace=True)

            # Extract the features and targets
            X = data.drop(columns=["Target"])
            y = data["Target"]

            return ProcessedData(X=X, y=y)
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )
