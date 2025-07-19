from core.config import config

from datetime import datetime
import pandas as pd


class PredictionStorage:

    def __init__(self, logger):
        self.base_dir = config.data.PREDICT_DATA_DIR
        self.logger = logger

    def load_prediction_csv(self, model_type: str, symbol: str, date: datetime):
        """
        Load prediction results from the corresponding CSV file for a specified date.

        Args:
            model_type (str): The type of model used for prediction (e.g., LSTM, Prophet).
            symbol (str): The stock symbol being predicted.
            date (datetime): Date associated to the prediction

        Returns:
            dict|None: The prediction results if Any, else None
        """

        # Generate the file path
        file_path = self.base_dir / model_type / f"{symbol}_predictions.csv"

        if not file_path.exists():
            return None

        # Load existing CSV
        df = pd.read_csv(file_path)

        # Convert the date to the format 'YYYY-MM-DD'
        day = date.date().isoformat()

        result = df[
            (df["symbol"] == symbol.upper())
            & (df["date"] == day)
            & (df["model_type"] == model_type)
        ]

        return result.iloc[0].to_dict() if not result.empty else None

    def get_existing_prediction_dates(self, model_type: str, symbol: str) -> list[str]:
        """
        Get a list of existing dates for which predictions have already been computed and stored
        in the csv file.

        Args:
            model_type (str): Model type (e.g., "lstm", "prophet").
            symbol (str): Stock symbol (e.g., "AAPL").

        Returns:
            list[str]: List of dates (in 'YYYY-MM-DD' format) that already have predictions.
        """

        # Generate the file path
        file_path = self.base_dir / model_type / f"{symbol}_predictions.csv"

        # If the CSV does not exist
        if not file_path.exists():
            return []

        # Load existing CSV
        df = pd.read_csv(file_path)

        # Extract the unique list of dates for which predictions have been made
        existing_dates = df["date"].unique().tolist()

        return existing_dates

    def save_prediction_to_csv(
        self,
        model_type: str,
        symbol: str,
        date: datetime,
        prediction: float,
        confidence: float,
        model_version: str,
    ):
        """
        Save the prediction results to a CSV file.

        Args:
            model_type (str): The type of model used for prediction (e.g., LSTM, Prophet).
            symbol (str): The stock symbol being predicted.
            date (datetime): Date associated to the prediction
            prediction (float): The predicted value from the model.
            confidence (float): The confidence score of the prediction.
            model_version (str): The version of the model used.
            file_path (str): Path to the CSV file where the results will be stored.
        """

        # Generate the file path
        file_path = self.base_dir / model_type / f"{symbol}_predictions.csv"

        # Make sure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        new_data = {
            "symbol": symbol.upper(),
            "date": date,
            "model_type": model_type,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
        }

        # Load existing CSV if exists
        if file_path.exists():
            df = pd.read_csv(file_path)

            # Drop any existing row for same symbol, date, and model_type
            df = df[
                ~(
                    (df["symbol"] == new_data["symbol"])
                    & (df["date"] == new_data["date"])
                    & (df["model_type"] == new_data["model_type"])
                )
            ]

            # Append new row
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        else:
            df = pd.DataFrame([new_data])

        # Save back
        df = df.sort_values(by=["date"])
        df.to_csv(file_path, index=False)

        self.logger.debug(
            f"Saved prediction result for model {model_type} for symbol {symbol} for date {date} to {file_path}"
        )
