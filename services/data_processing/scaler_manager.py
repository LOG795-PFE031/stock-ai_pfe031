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

    def __init__(self, model_type: str, symbol: str):
        self.model_type = model_type
        self.symbol = symbol

    def model_requires_scaling(self) -> bool:
        return ScalerFactory.model_requires_scaling(self.model_type)

    def create_scaler(self) -> MinMaxScaler:
        return ScalerFactory.create_scaler(self.model_type)

    def save_scaler(self, scaler, phase, scaler_type: str):
        """
        Save the given scaler to disk.

        Args:
            scaler: A fitted scikit-learn scaler to be saved.
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
        """
        try:
            joblib.dump(scaler, self.get_scaler_path(phase, scaler_type))
        except Exception as e:
            raise e

    def load_scaler(
        self, phase: str, scaler_type: str = FEATURES_SCALER_TYPE
    ) -> MinMaxScaler:
        """
        Load the saved scaler from disk given the model_type and the symbol

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.

        Returns:
            Any (sklearn scalers): The loaded scaler instance.
        """
        scaler_path = self.get_scaler_path(phase, scaler_type)

        if not scaler_path.exists:
            raise FileNotFoundError(
                f"Scaler not found for model {self.model_type} for symbol {self.symbol} for {phase} phase"
            )

        return joblib.load(scaler_path)

    def promote_scaler(self) -> bool:
        """
        Promote scalers from the 'training' phase to the 'prediction' phase (if needed).

        Returns:
            bool: True if the scalers were successfully promoted, False if no promotion was needed.
        """
        try:
            if self.model_requires_scaling():

                # Get the training scalers
                src_features_scaler_path = self.get_scaler_path(
                    "training", self.FEATURES_SCALER_TYPE
                )
                src_targets_scaler_path = self.get_scaler_path(
                    "training", self.TARGETS_SCALER_TYPE
                )

                # Switch the phase to get the prediction scalers paths
                self.phase = "prediction"

                dst_features_scaler_path = self.get_scaler_path(
                    "prediction", self.FEATURES_SCALER_TYPE
                )
                dst_targets_scaler_path = self.get_scaler_path(
                    "prediction", self.TARGETS_SCALER_TYPE
                )

                if (
                    not src_features_scaler_path.exists()
                    or not src_targets_scaler_path.exists()
                ):
                    raise FileNotFoundError(f"Missing scaler to promote")

                # Copy the training scalers to prediction scalers
                dst_features_scaler_path.write_bytes(
                    src_features_scaler_path.read_bytes()
                )
                dst_targets_scaler_path.write_bytes(
                    src_targets_scaler_path.read_bytes()
                )

                return True
            else:
                return False
        except Exception as e:
            raise RuntimeError("Error while promoting the scalers") from e

    def get_scaler_path(
        self, phase: str, scaler_type: str = FEATURES_SCALER_TYPE
    ) -> Path:
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
            config.preprocessing.SCALERS_DIR / self.model_type / phase / self.symbol
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
