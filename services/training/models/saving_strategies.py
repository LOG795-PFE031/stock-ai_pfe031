from abc import ABC, abstractmethod


class BaseSaver(ABC):
    @abstractmethod
    def save_model(self, model, path):
        """Save the trained model to disk."""
        pass

    async def save(self, model, path):
        """Save the trained model to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.save_model(model, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to '{path}': {e}") from e


class JoblibSaver(BaseSaver):
    def save_model(self, model, path):
        import joblib

        joblib.dump(model, path)


class KerasSaver(BaseSaver):
    def save_model(self, model, path):
        model.save(str(path))
