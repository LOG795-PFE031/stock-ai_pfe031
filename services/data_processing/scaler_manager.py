from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler

from core.config import config


class ScalerFactory:
    _SCALABLE_MODELS = {"lstm"}

    @staticmethod
    def create_scaler(model_type: str) -> MinMaxScaler:
        if model_type in ScalerFactory._SCALABLE_MODELS:
            return MinMaxScaler()
        return None

    @staticmethod
    def model_requires_scaling(model_type: str) -> bool:
        return model_type in ScalerFactory._SCALABLE_MODELS


class ScalerManager:
    FEATURES_SCALER_TYPE = "features"
    TARGETS_SCALER_TYPE = "targets"
    VALID_SCALER_TYPES = {FEATURES_SCALER_TYPE, TARGETS_SCALER_TYPE}

    def __init__(self, model_type: str, symbol: str, phase: str):
        self.model_type = model_type
        self.symbol = symbol
        self.phase = phase

    def model_requires_scaling(self) -> bool:
        return ScalerFactory.model_requires_scaling(self.model_type)

    def create_scaler(self) -> MinMaxScaler:
        return ScalerFactory.create_scaler(self.model_type)

    def save_scaler(self, scaler, scaler_type: str):
        """
        Save the given scaler to disk.

        Args:
            scaler: A fitted scikit-learn scaler to be saved.
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
        """
        try:
            joblib.dump(scaler, self.get_scaler_path(scaler_type))
        except Exception as e:
            raise e

    def load_scaler(self, scaler_type: str = FEATURES_SCALER_TYPE) -> MinMaxScaler:
        """
        Load the saved scaler from disk given the model_type and the symbol

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.

        Returns:
            Any (sklearn scalers): The loaded scaler instance.
        """
        scaler_path = self.get_scaler_path(scaler_type)

        if not scaler_path.exists:
            raise FileNotFoundError(
                f"Scaler not found for model {self.model_type} for symbol {self.symbol}"
            )

        return joblib.load(scaler_path)

    def get_scaler_path(self, scaler_type: str = FEATURES_SCALER_TYPE) -> Path:
        """
        Returns the full path to the scaler file based on model type, symbol, and phase.
        Creates the directory if it doesn't exist.

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.

        Returns:
            Path: Path to the scaler file.
        """

        self._validate_scaler_type(scaler_type)

        # Directory of the scaler
        scaler_dir = (
            config.preprocessing.SCALERS_DIR
            / self.model_type
            / self.phase
            / self.symbol
        )
        scaler_dir.mkdir(parents=True, exist_ok=True)

        # Full scaler path (with the corresponding phase)
        scaler_path = scaler_dir / f"{scaler_type}_scaler.gz"

        return scaler_path

    def _validate_scaler_type(self, scaler_type: str):
        """
        Validates the scaler type to ensure it is either 'features' or 'targets'.

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.
        """
        if scaler_type not in self.VALID_SCALER_TYPES:
            raise ValueError(
                f"Invalid scaler_type '{scaler_type}'. Must be one of {self.VALID_SCALER_TYPES}"
            )
