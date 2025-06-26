from abc import ABC, abstractmethod
from pathlib import Path


class BaseSaver(ABC):
    @abstractmethod
    def save_model(self, model, base_path) -> Path:
        """Save the trained model to disk."""
        pass

    async def save(self, model, base_path) -> Path:
        """Save the trained model to disk."""
        try:
            base_path.mkdir(parents=True, exist_ok=True)
            model_path = self.save_model(model, base_path)
            return model_path
        except Exception as e:
            raise RuntimeError(f"Failed to save model to '{base_path}': {e}") from e


class JoblibSaver(BaseSaver):
    def save_model(self, model, base_path):
        import joblib

        model_path = base_path / "model.joblib"
        joblib.dump(model, model_path)

        return model_path


class KerasSaver(BaseSaver):
    def save_model(self, model, base_path):
        model_path = base_path / "model.keras"
        model.save(str(model_path))
        return model_path
