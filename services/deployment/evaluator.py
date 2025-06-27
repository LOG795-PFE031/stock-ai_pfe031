import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.types import Metrics
from typing import Any, Tuple


class Evaluator:
    def evaluate(self, y_true, y_pred) -> Metrics:
        """y_true = y_true[:, 2]  # Close price is the third column
        y_pred = y_pred.ravel()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)"""
        pass

    '''async def evaluate(
        self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Metrics:
        """Evaluate LSTM model."""
        try:
            X_test, y_test = test_data

            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            y_test = y_test[:, 2]  # Close price is the third column
            y_pred = y_pred.ravel()

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            return Metrics(
                mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2)
            )

        except Exception as e:
            self.lo
            
        async def evaluate(self, model: Prophet, test_data: pd.DataFrame) -> Metrics:
        """Evaluate Prophet model."""
        try:
            # Make predictions
            forecast = model.predict(test_data)

            # Calculate metrics
            y_true = test_data["y"].values
            y_pred = forecast["yhat"].values

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            return Metrics(
                mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2)
            )

        except Exception as e:
            self.logger.er'''
