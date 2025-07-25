from mlflow.pyfunc import PythonModel


class XGBoostPredictor(PythonModel):
    def load_context(self, context):
        import joblib

        model_path = context.artifacts.get("model")
        if not model_path:
            raise ValueError(
                "Model path for the XGBoost model is missing from MLflow artifacts."
            )
        self.model = joblib.load(model_path)

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)
