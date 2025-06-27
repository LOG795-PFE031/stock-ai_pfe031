from mlflow.pyfunc import PyFuncModel
from typing import Union
import pandas as pd
import numpy as np


class Predictor:
    def predict(
        self, model: PyFuncModel, X: Union[pd.DataFrame, np.ndarray, list]
    ) -> Union[np.ndarray, pd.Series, list]:
        """
        Perform prediction using a loaded MLflow PyFunc model.


        Args:
            model (PyFuncModel): A MLflow pyfunc model
            X (Union[pd.DataFrame, np.ndarray, list]): Input data for prediction

        Returns:
            Union[np.ndarray, pd.Series, list]: The model predictions
        """
        y = model.predict(X)
        return y
