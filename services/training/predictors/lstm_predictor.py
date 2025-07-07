from mlflow.pyfunc import PythonModel


class LSTMPredictor(PythonModel):
    def load_context(self, context):
        from keras import models  # Replace tensorflow import

        model_path = context.artifacts.get("model")
        if not model_path:
            raise ValueError(
                f"Model path for {self.model_name} is missing from MLflow artifacts."
            )

        self.model = models.load_model(model_path, compile=False)

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)
