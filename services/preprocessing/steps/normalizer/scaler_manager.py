import joblib

from core.config import config


class ScalerManager:
    def __init__(self, model_type: str, symbol: str, phase: str):
        self.model_type = model_type
        self.symbol = symbol
        self.phase = phase

    def save_scaler(self, scaler):
        """
        Save the given scaler to disk.

        Args:
            scaler: A fitted scikit-learn scaler to be saved.
        """
        try:
            joblib.dump(scaler, self._get_scaler_path())
        except Exception as e:
            raise e

    def load_scaler(self):
        """
        Load the saved scaler from disk given the model_type and the symbol

        Returns:
            Any (sklearn scalers): The loaded scaler instance.
        """
        scaler_path = self._get_scaler_path()

        if not scaler_path.exists:
            raise FileNotFoundError(
                f"Scaler not found for model {self.model_type} for symbol {self.symbol}"
            )

        return joblib.load(scaler_path)

    def _get_scaler_path(self):
        # Directory of the scaler
        scaler_dir = config.preprocessing.SCALERS_DIR / self.model_type / self.symbol
        scaler_dir.mkdir(parents=True, exist_ok=True)

        # Full scaler path (with the corresponding phase)
        scaler_path = scaler_dir / f"{self.phase}_scaler.gz"

        return scaler_path
