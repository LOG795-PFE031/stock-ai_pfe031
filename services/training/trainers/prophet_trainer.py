from prophet import Prophet
import pandas as pd
from typing import Any, Tuple, Dict

from .base_trainer import BaseTrainer
from core.types import ProphetInput


class ProphetTrainer(BaseTrainer):
    async def train(
        self, data: ProphetInput, **kwargs
    ) -> Tuple[Prophet, Dict[str, Any]]:
        try:
            # Extract the features (only X, since the Prophet model is trained using a DataFrame)
            data = pd.DataFrame(data.X)

            # Initialize Prophet model
            model = Prophet(
                changepoint_prior_scale=kwargs.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=kwargs.get("seasonality_prior_scale", 10.0),
                holidays_prior_scale=kwargs.get("holidays_prior_scale", 10.0),
                seasonality_mode=kwargs.get("seasonality_mode", "multiplicative"),
            )

            # Add additional regressors
            for feature in data.columns:
                if feature not in ["Close", "Date", "ds", "y"]:
                    model.add_regressor(feature)

            # Fit model
            model.fit(data)

            # Get training history
            history = {
                "changepoints": (
                    model.changepoints.tolist()
                    if hasattr(model, "changepoints")
                    else []
                ),
                "trend": model.params["k"].tolist() if "k" in model.params else [],
                "seasonality": (
                    model.params["beta"].tolist() if "beta" in model.params else []
                ),
            }

            return model, history

        except Exception as e:
            raise RuntimeError(
                f"Error occurred while training the Prophet model: {str(e)}"
            ) from e
