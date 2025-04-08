"""
Prophet model trainer.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

from training.base_trainer import BaseTrainer
from core.config import config
from core.logging import logger

class ProphetTrainer(BaseTrainer):
    """Trainer for Prophet models."""
    
    def __init__(self):
        super().__init__("prophet")
    
    async def prepare_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data for Prophet model."""
        try:
            # Load stock data
            data_file = self.config.data.STOCK_DATA_DIR / f"{symbol}_data.csv"
            df = pd.read_csv(data_file)
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter date range if specified
            if start_date:
                df = df[df['Date'] >= start_date]
            if end_date:
                df = df[df['Date'] <= end_date]
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df['Date'],
                'y': df['Close']
            })
            
            # Add additional regressors
            for feature in self.config.model.FEATURES:
                if feature not in ['Close', 'Date']:
                    prophet_df[feature] = df[feature]
            
            # Split into train and test
            train_size = int(len(prophet_df) * 0.8)
            train_data = prophet_df[:train_size]
            test_data = prophet_df[train_size:]
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    async def train(
        self,
        symbol: str,
        data: pd.DataFrame,
        **kwargs
    ) -> Tuple[Prophet, Dict[str, Any]]:
        """Train Prophet model."""
        try:
            # Initialize Prophet model
            model = Prophet(
                changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=kwargs.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=kwargs.get('holidays_prior_scale', 10.0),
                seasonality_mode=kwargs.get('seasonality_mode', 'multiplicative')
            )
            
            # Add additional regressors
            for feature in self.config.model.FEATURES:
                if feature not in ['Close', 'Date']:
                    model.add_regressor(feature)
            
            # Fit model
            model.fit(data)
            
            # Get training history
            history = {
                "changepoints": model.changepoints.tolist(),
                "trend": model.trend.tolist(),
                "seasonality": model.seasonality.tolist()
            }
            
            return model, history
            
        except Exception as e:
            self.logger.error(f"Error training Prophet model for {symbol}: {str(e)}")
            raise
    
    async def evaluate(
        self,
        model: Prophet,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate Prophet model."""
        try:
            # Make predictions
            forecast = model.predict(test_data)
            
            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = forecast['yhat'].values
            
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(mse)
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating Prophet model: {str(e)}")
            raise
    
    async def save_model(
        self,
        model: Prophet,
        symbol: str,
        metrics: Dict[str, float]
    ) -> None:
        """Save Prophet model."""
        try:
            # Create prophet directory if it doesn't exist
            prophet_dir = self.model_dir / "prophet"
            prophet_dir.mkdir(parents=True, exist_ok=True)
            
            # Create symbol-specific directory
            symbol_dir = prophet_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model using joblib
            model_path = symbol_dir / f"{symbol}_model.joblib"
            joblib.dump(model, model_path)
            
            # Save metrics with timestamp
            metrics["timestamp"] = datetime.now().isoformat()
            metrics["model_version"] = self.model_version
            metrics_path = symbol_dir / f"{symbol}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"Saved Prophet model for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving Prophet model for {symbol}: {str(e)}")
            raise 