from typing import Any, Tuple, Dict

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from core.types import XGBoostInput
from .base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    async def train(
        self, data: XGBoostInput, **kwargs
    ) -> Tuple[XGBRegressor, Dict[str, Any]]:
        """Train the XGBoost model"""
        try:
            # Extract the features and targets
            X_train = pd.DataFrame(data.X)
            y_train = np.array(data.y)

            # Initialize XGBoost model
            model = XGBRegressor(n_estimators=5, max_depth=3)

            # Fit model
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)])

            # Get training history
            history = model.evals_result()

            return model, history

        except Exception as exception:
            raise RuntimeError(
                f"Error occurred while training the XGBoost model: {str(exception)}"
            ) from exception
