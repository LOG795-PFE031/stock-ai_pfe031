"""
Prophet model implementation for stock prediction.
"""
import os
import json
import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

from .base import BaseModel

class ProphetModel(BaseModel):
    """Prophet model for stock price prediction"""
    
    def __init__(self, config: Any):
        """
        Initialize the Prophet model
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.features = config.model.FEATURES
        self.epochs = config.model.EPOCHS
        self.logger = logging.getLogger('ProphetModel')
        
        # Prophet model parameters with advanced configuration
        self.params = {
            'changepoint_prior_scale': 0.05,  # Controls flexibility of trend
            'seasonality_prior_scale': 10,    # Controls flexibility of seasonality
            'seasonality_mode': 'multiplicative',  # Better for stock prices
            'daily_seasonality': True,        # Capture intraday patterns
            'weekly_seasonality': True,       # Capture weekly patterns
            'yearly_seasonality': True,       # Capture yearly patterns
            'holidays_prior_scale': 10,       # Controls flexibility of holiday effects
            'changepoint_range': 0.9,         # Allow trend changes in 90% of data
            'interval_width': 0.95,           # 95% confidence interval
            'mcmc_samples': 0,                # Disable MCMC for faster training
            'stan_backend': 'CMDSTANPY'       # Use CmdStanPy for better performance
        }
    
    def build(self) -> None:
        """Build the Prophet model"""
        self.model = Prophet(**self.params)
    
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       **kwargs) -> pd.DataFrame:
        """
        Preprocess data for Prophet model with advanced features
        
        Args:
            data: Input DataFrame
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data for Prophet model")
        
        # Prophet requires 'ds' and 'y' columns
        df = data.copy()
        df['ds'] = pd.to_datetime(df['Date'])
        df['y'] = df['Close']
        
        # Add technical indicators as regressors
        if 'Volume' in df.columns:
            df['volume'] = df['Volume'].fillna(0)
            self.model.add_regressor('volume')
            self.logger.info("Added volume as regressor")
        
        if 'RSI' in df.columns:
            df['rsi'] = df['RSI'].fillna(df['RSI'].mean())
            self.model.add_regressor('rsi')
            self.logger.info("Added RSI as regressor")
        
        if 'MACD' in df.columns:
            df['macd'] = df['MACD'].fillna(df['MACD'].mean())
            self.model.add_regressor('macd')
            self.logger.info("Added MACD as regressor")
        
        if 'MACD_Signal' in df.columns:
            df['macd_signal'] = df['MACD_Signal'].fillna(df['MACD_Signal'].mean())
            self.model.add_regressor('macd_signal')
            self.logger.info("Added MACD Signal as regressor")
        
        # Add volatility as regressor
        if 'Volatility' in df.columns:
            df['volatility'] = df['Volatility'].fillna(df['Volatility'].mean())
            self.model.add_regressor('volatility')
            self.logger.info("Added volatility as regressor")
        
        # Add moving averages as regressors
        if 'MA_5' in df.columns:
            df['ma_5'] = df['MA_5'].fillna(df['MA_5'].mean())
            self.model.add_regressor('ma_5')
            self.logger.info("Added 5-day MA as regressor")
        
        if 'MA_20' in df.columns:
            df['ma_20'] = df['MA_20'].fillna(df['MA_20'].mean())
            self.model.add_regressor('ma_20')
            self.logger.info("Added 20-day MA as regressor")
        
        # Add returns as regressor
        if 'Returns' in df.columns:
            df['returns'] = df['Returns'].fillna(0)
            self.model.add_regressor('returns')
            self.logger.info("Added returns as regressor")
        
        # Add custom seasonality
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        self.logger.info("Added monthly seasonality")
        
        # Add custom seasonality for earnings seasons
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=5
        )
        self.logger.info("Added quarterly seasonality")
        
        return df
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the Prophet model with advanced settings
        
        Args:
            train_data: Training data
            val_data: Optional validation data (not used for Prophet)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        self.logger.info("Starting Prophet model training")
        
        # Preprocess data
        df = self.preprocess_data(train_data)
        
        # Fit model
        self.model.fit(df)
        
        # Save model parameters and last data
        model_data = {
            'params': self.params,
            'regressors': self.model.extra_regressors.keys(),
            'seasonalities': [s.name for s in self.model.seasonalities.values()],
            'last_data': df.tail(60).to_dict(orient='records'),
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'data_points': len(df),
                'data_start': df['ds'].min().strftime('%Y-%m-%d'),
                'data_end': df['ds'].max().strftime('%Y-%m-%d'),
                'regressors': list(self.model.extra_regressors.keys()),
                'seasonalities': [s.name for s in self.model.seasonalities.values()]
            }
        }
        
        self.logger.info("Prophet model training completed successfully")
        return {
            'model_data': model_data,
            'model': self.model
        }
    
    def predict(self, 
                data: pd.DataFrame,
                periods: int = 1,
                **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data for prediction
            periods: Number of periods to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe if they exist
        for regressor in self.model.extra_regressors.keys():
            if regressor in data.columns:
                future[regressor] = data[regressor].iloc[-1]
            else:
                self.logger.warning(f"Regressor {regressor} not found in input data")
        
        # Make prediction
        forecast = self.model.predict(future)
        
        # Return only the predicted values
        return forecast['yhat'].values[-periods:]
    
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path where to save the model
        """
        # Save model parameters and last data
        model_data = {
            'params': self.params,
            'regressors': self.model.extra_regressors.keys(),
            'seasonalities': [s.name for s in self.model.seasonalities.values()],
            'last_data': self.model.history.tail(60).to_dict(orient='records'),
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'data_points': len(self.model.history),
                'data_start': self.model.history['ds'].min().strftime('%Y-%m-%d'),
                'data_end': self.model.history['ds'].max().strftime('%Y-%m-%d'),
                'regressors': list(self.model.extra_regressors.keys()),
                'seasonalities': [s.name for s in self.model.seasonalities.values()]
            }
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path from where to load the model
        """
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Create new model with saved parameters
        self.params = model_data['params']
        self.model = Prophet(**self.params)
        
        # Add regressors
        for regressor in model_data['regressors']:
            self.model.add_regressor(regressor)
        
        # Add seasonalities
        for seasonality in model_data['seasonalities']:
            if seasonality == 'monthly':
                self.model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )
            elif seasonality == 'quarterly':
                self.model.add_seasonality(
                    name='quarterly',
                    period=91.25,
                    fourier_order=5
                )
        
        # Convert last data back to DataFrame
        df = pd.DataFrame(model_data['last_data'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Fit the model on the last data
        self.model.fit(df)
        
        self.logger.info(f"Model loaded from {path}")
    
    def evaluate(self, 
                 test_data: pd.DataFrame,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        predictions = self.predict(test_data, periods=len(test_data))
        
        # Calculate metrics
        actual = test_data['Close'].values
        mae = np.mean(np.abs(predictions - actual))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        r2 = 1 - np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        }
        
        self.logger.info(f"Model evaluation metrics: {metrics}")
        return metrics 