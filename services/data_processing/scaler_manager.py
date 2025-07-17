from pathlib import Path
import joblib
import json
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

    # Path to the scaler registry (JSON file)
    REGISTRY_PATH = config.preprocessing.SCALER_REGISTRY_JSON

    # Scaler types
    FEATURES_SCALER_TYPE = "features"
    TARGETS_SCALER_TYPE = "targets"
    VALID_SCALER_TYPES = {FEATURES_SCALER_TYPE, TARGETS_SCALER_TYPE}

    # Phases
    TRAINING_PHASE = "training"
    PREDICTION_PHASE = "prediction"
    VALID_PHASES = [PREDICTION_PHASE, TRAINING_PHASE]

    def __init__(self, model_type: str, symbol: str):
        self.model_type = model_type
        self.symbol = symbol
        self.registry = self._load_registry()

    def model_requires_scaling(self) -> bool:
        return ScalerFactory.model_requires_scaling(self.model_type)

    def create_scaler(self) -> MinMaxScaler:
        return ScalerFactory.create_scaler(self.model_type)

    def save_scaler(self, scaler, scaler_type: str, scaler_dates: tuple[str, str]):
        """
        Save the given scaler to disk.

        Args:
            scaler: A fitted scikit-learn scaler to be saved.
            scaler_type (str): The type of scaler. Must be one of 'features' or 'targets'.
            scaler_dates tuple[str,str]: Start and end dates used in the fitting of the scaler.
        """
        try:
            if not scaler_dates:
                raise ValueError("You must provide scaler_dates when saving a scaler.")

            # Get the save path of the scaler
            save_path = self.get_scaler_path(
                scaler_type=scaler_type, scaler_dates=scaler_dates
            )

            # Save the sacler
            joblib.dump(scaler, save_path)

            # Retrieves the entry to the scaler root path in the registry
            scaler_root_path = (
                self.registry.setdefault(self.model_type, {})
                .setdefault(self.symbol, {})
                .setdefault(scaler_type, {})
            )

            # Generate the date identifier (key)
            key = self._generate_date_identifier(scaler_dates=scaler_dates)

            # Add the save path to the scaler registry (history)
            scaler_root_path.setdefault("history", {})[key] = str(save_path)

            # Update the training pointer of the registry with the saved path
            scaler_root_path["training"] = str(save_path)

            # Save the registry
            self._save_registry()
        except Exception as e:
            raise e

    def load_scaler(
        self,
        scaler_type: str,
        phase: str = PREDICTION_PHASE,
        scaler_dates: tuple[str, str] = None,
    ) -> MinMaxScaler:
        """
        Load a saved scaler from disk using the registry based on provided parameters.

        If `scaler_dates` is provided, the method will look up the scaler in the historical registry
        using the specified date range. Otherwise, it will look it up based on the current phase
        (e.g. 'training' or 'prediction').

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.
            phase (str, optional): The data processing phase. Defaults to PREDICTION_PHASE.
                Must be one of the valid phases defined in `self.VALID_PHASES`. Ignored if `scaler_dates` is provided.
            scaler_dates (tuple[str, str], optional): Tuple of (start_date, end_date).
                If provided, loads a historical scaler for the specific date range.

        Returns:
            Any (sklearn scalers): The loaded scaler instance.
        """
        if not scaler_dates:

            if phase not in self.VALID_PHASES:
                raise ValueError(
                    f"Invalid phase '{phase}'. Must be one of {self.VALID_PHASES}."
                )

            scaler_path = self.registry[self.model_type][self.symbol][scaler_type][
                phase
            ]

        else:
            date_identifier = self._generate_date_identifier(scaler_dates=scaler_dates)
            scaler_path = self.registry[self.model_type][self.symbol][scaler_type][
                "history"
            ][date_identifier]

        # Turn the string path to a Path
        scaler_path = Path(scaler_path)

        if not scaler_path.exists:
            raise FileNotFoundError(
                f"Scaler not found for model {self.model_type} for symbol {self.symbol} for the date range {scaler_dates} phase"
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

                # Targets scaler path to promote
                targets_scaler_path_to_promote = Path(
                    self.registry[self.model_type][self.symbol][
                        self.TARGETS_SCALER_TYPE
                    ][self.TRAINING_PHASE]
                )

                # Features scaler path to promote
                features_scaler_path_to_promote = Path(
                    self.registry[self.model_type][self.symbol][
                        self.FEATURES_SCALER_TYPE
                    ][self.TRAINING_PHASE]
                )

                if (
                    not features_scaler_path_to_promote.exists()
                    or not targets_scaler_path_to_promote.exists()
                ):
                    raise FileNotFoundError(f"Missing scaler to promote")

                # Set the 'prediction' pointers for target and features
                self._set_prediction_scaler_path(
                    self.FEATURES_SCALER_TYPE, features_scaler_path_to_promote
                )
                self._set_prediction_scaler_path(
                    self.TARGETS_SCALER_TYPE, targets_scaler_path_to_promote
                )

                # Save the registry updates
                self._save_registry()

                return True
            else:
                return False
        except Exception as e:
            raise RuntimeError("Error while promoting the scalers") from e

    def get_scaler_path(self, scaler_type: str, scaler_dates: tuple[str, str]) -> Path:
        """
        Returns the full path to the scaler file based on model type, symbol, and scaler dates.
        Creates the directory if it doesn't exist.

        Args:
            scaler_type: The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.
            scaler_dates (tuple[str, str]): Tuple of (start_date, end_date). Dates used to fit the scaler

        Returns:
            Path: Path to the scaler file.
        """

        self._validate_scaler_type(scaler_type)

        # Generate the date identifier (used in the scaler filename)
        date_identifier = self._generate_date_identifier(scaler_dates=scaler_dates)

        # Directory of the scaler
        scaler_dir = (
            config.preprocessing.SCALERS_DIR
            / self.model_type
            / self.symbol
            / date_identifier
            / scaler_type
        )
        scaler_dir.mkdir(parents=True, exist_ok=True)

        # Full scaler path (with the corresponding phase)
        scaler_path = scaler_dir / "scaler.gz"

        return scaler_path

    def _load_registry(self):
        """Load the json registry"""
        if not self.REGISTRY_PATH.exists():
            return {}
        with open(self.REGISTRY_PATH, "r") as f:
            return json.load(f)

    def _save_registry(self):
        """Save the registry to local disk"""
        self.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.REGISTRY_PATH, "w") as f:
            json.dump(self.registry, f, indent=4)

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

    def _set_prediction_scaler_path(self, scaler_type: str, path: Path):
        """
        Set the prediction scaler path (main scaler used in Production)

        Args:
            scaler_type (str): The type of scaler. Must be one of 'features' or 'targets'.
                This determines whether the scaler is for input features or the targets variable.
            path (Path): Path to set
        """
        self.registry.setdefault(self.model_type, {}).setdefault(
            self.symbol, {}
        ).setdefault(scaler_type, {})["prediction"] = str(path)

    def _generate_date_identifier(self, scaler_dates: tuple[str, str]) -> str:
        """
        Generate a date identifier used in the registry and filename of a saved scaler

        Args:
            scaler_dates (tuple[str, str]): Tuple of (start_date, end_date). Dates used to fit the scaler
        """

        start_date = scaler_dates[0]
        end_date = scaler_dates[1]
        date_identifier = f"{start_date}_{end_date}"

        return date_identifier
