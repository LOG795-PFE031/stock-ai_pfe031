from prophet import Prophet
import pandas as pd
from typing import Any, Tuple, Dict

from .base_trainer import BaseTrainer
from core.types import FormattedInput


class ProphetTrainer(BaseTrainer):
    async def train(
        self, data: FormattedInput[pd.DataFrame, pd.DataFrame], **kwargs
    ) -> Tuple[Prophet, Dict[str, Any]]:
        try:
            # Extract the features (only X, since the Prophet model is trained using a DataFrame)
            data = data.X

            # Initialize Prophet model
            model = Prophet(
                changepoint_prior_scale=kwargs.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=kwargs.get("seasonality_prior_scale", 10.0),
                holidays_prior_scale=kwargs.get("holidays_prior_scale", 10.0),
                seasonality_mode=kwargs.get("seasonality_mode", "multiplicative"),
            )

            # Add additional regressors
            for feature in data.X.columns:
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
            self.logger.error(
                f"Error training Prophet model for {self.symbol}: {str(e)}"
            )
            raise
