"""
Stock prediction service.
"""
from typing import Dict, Any, Optional, List
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random

from services.base_service import BaseService
from core.utils import (
    calculate_technical_indicators,
    validate_stock_symbol,
    format_prediction_response
)
from core.logging import logger
from services.model_service import ModelService
from services.data_service import DataService

def configure_tensorflow():
    """Configure TensorFlow for CPU-only operation."""
    # Configure TensorFlow for CPU-only
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    logger['prediction'].info("TensorFlow configured for CPU-only operation")
    return False  # Always return False since we're using CPU-only version

# Configure TensorFlow at module level
USE_GPU = configure_tensorflow()

class PredictionService(BaseService):
    """Service for stock price predictions."""
    
    def __init__(self, model_service: ModelService, data_service: DataService):
        super().__init__()
        self.models = {}
        self.scalers = {}
        self.model_version = "0.1.0"
        self._initialized = False
        self.use_gpu = False  # Always False since we're using CPU-only version
        self.model_service = model_service
        self.data_service = data_service
        self.logger = logger['prediction']
    
    async def initialize(self) -> None:
        """Initialize the prediction service."""
        try:
            # Load models and scalers
            await self._load_models()
            self._initialized = True
            self.logger.info("Prediction service initialized successfully with CPU configuration")
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction service: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear models from memory
            self.models.clear()
            self.scalers.clear()
            self._initialized = False
            
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            self.logger.info("Prediction service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during prediction service cleanup: {str(e)}")
    
    async def _load_models(self) -> None:
        """Load prediction models and scalers."""
        # Implementation will be added later
        pass
    
    async def get_prediction(
        self,
        symbol: str,
        model_type: str = "lstm"
    ) -> Dict[str, Any]:
        """
        Get stock price prediction for a given symbol.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model to use (lstm or prophet)
            
        Returns:
            Dictionary containing prediction results
        """
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        if model_type not in ["lstm", "prophet"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            # Get prediction based on model type
            if model_type == "lstm":
                prediction = await self._get_lstm_prediction(symbol)
            else:
                prediction = await self._get_prophet_prediction(symbol)
            
            # Format the response using the utility function
            return format_prediction_response(
                prediction=prediction["prediction"],
                confidence=prediction["confidence_score"],
                model_type=prediction["model_type"],
                model_version=prediction["model_version"]
            )
            
        except Exception as e:
            self.logger.error(f"Error getting prediction for {symbol}: {str(e)}")
            raise
    
    async def _get_lstm_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get prediction using LSTM model."""
        try:
            # Load the latest model and scaler
            model_result = await self.model_service.load_model(symbol, "lstm")
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load LSTM model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            # Get latest data
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            # Prepare features
            df = data_result["data"]
            features = df[self.config.model.FEATURES].values
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Create sequence for LSTM (samples, time steps, features)
            sequence = scaled_features[-self.config.model.SEQUENCE_LENGTH:].reshape(
                1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES)
            )
            
            # Make prediction
            scaled_prediction = model.predict(sequence)
            
            # Get the last actual close price for scaling reference
            last_close = df['Close'].iloc[-1]
            last_scaled = scaled_features[-1, self.config.model.FEATURES.index('Close')]
            
            # Inverse transform the prediction
            prediction = self._inverse_scale_lstm_prediction(
                scaled_prediction[0, 0],
                last_close,
                last_scaled,
                scaler,
                self.config.model.FEATURES.index('Close')
            )
            
            # Calculate confidence based on prediction variance
            confidence = self._calculate_lstm_confidence(
                sequence,
                scaled_prediction,
                model,
                scaler,
                self.config.model.FEATURES.index('Close')
            )
            
            # Prepare prediction details
            prediction_details = {
                "last_close": float(last_close),
                "predicted_change": float(prediction - last_close),
                "predicted_change_percent": float((prediction - last_close) / last_close * 100),
                "sequence_length": self.config.model.SEQUENCE_LENGTH,
                "features_used": self.config.model.FEATURES
            }
            
            return {
                "prediction": float(prediction),
                "timestamp": datetime.now() + timedelta(days=1),
                "confidence_score": float(confidence),
                "model_version": model_result.get("version", "1.0.0"),
                "model_type": "lstm",
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            self.logger.error(f"Error getting LSTM prediction for {symbol}: {str(e)}")
            raise

    def _inverse_scale_lstm_prediction(
        self,
        scaled_prediction: float,
        last_close: float,
        last_scaled: float,
        scaler: Any,
        close_idx: int
    ) -> float:
        """Inverse scale the LSTM prediction."""
        # Calculate the predicted change in scaled space
        scaled_change = scaled_prediction - last_scaled
        
        # Create a dummy array for inverse transform
        dummy = np.zeros((1, len(self.config.model.FEATURES)))
        dummy[0, close_idx] = scaled_change
        
        # Inverse transform the change
        real_change = scaler.inverse_transform(dummy)[0, close_idx]
        
        # Apply the change to the last actual price
        return last_close + real_change

    def _calculate_lstm_confidence(
        self,
        sequence: np.ndarray,
        prediction: np.ndarray,
        model: Any,
        scaler: Any,
        close_idx: int
    ) -> float:
        """Calculate confidence score for LSTM prediction."""
        # Generate multiple predictions with dropout enabled
        predictions = []
        for _ in range(10):  # Monte Carlo dropout
            pred = model(sequence, training=True)
            predictions.append(pred.numpy()[0, 0])
        
        # Calculate prediction variance
        variance = np.var(predictions)
        
        # Scale variance to confidence score (0 to 1)
        # Higher variance means lower confidence
        confidence = np.exp(-variance)
        
        return float(np.clip(confidence, 0, 1))
    
    async def _get_prophet_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get prediction using Prophet model."""
        try:
            # Load the latest model
            model_result = await self.model_service.load_model(symbol, "prophet")
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load Prophet model for {symbol}")
            
            model = model_result["model"]
            
            # Get latest data
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            # Prepare data for Prophet
            df = data_result["data"]
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Date']),
                'y': df['Close']
            })
            
            # Add regressors if available
            for feature in self.config.model.FEATURES:
                if feature not in ['Close', 'Date']:
                    prophet_df[feature] = df[feature]
                    if not hasattr(model, feature):
                        model.add_regressor(feature)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=1)
            
            # Add latest regressor values to future dataframe
            for feature in self.config.model.FEATURES:
                if feature not in ['Close', 'Date']:
                    future[feature] = df[feature].iloc[-1]
            
            # Make prediction
            forecast = model.predict(future)
            prediction = forecast.iloc[-1]['yhat']
            
            # Calculate confidence using Prophet's uncertainty intervals
            lower = forecast.iloc[-1]['yhat_lower']
            upper = forecast.iloc[-1]['yhat_upper']
            interval_width = upper - lower
            last_price = df['Close'].iloc[-1]
            confidence = max(0, min(1, 1 - (interval_width / (4 * last_price))))
            
            # Prepare prediction details
            prediction_details = {
                "last_close": float(last_price),
                "predicted_change": float(prediction - last_price),
                "predicted_change_percent": float((prediction - last_price) / last_price * 100),
                "confidence_interval": {
                    "lower": float(lower),
                    "upper": float(upper)
                }
            }
            
            return {
                "prediction": float(prediction),
                "timestamp": datetime.now() + timedelta(days=1),
                "confidence_score": float(confidence),
                "model_version": model_result.get("version", "1.0.0"),
                "model_type": "prophet",
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Prophet prediction for {symbol}: {str(e)}")
            raise
    
    async def get_historical_predictions(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get historical predictions for a given symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical predictions
            
        Returns:
            Dictionary containing historical predictions
        """
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        try:
            # Implementation will be added later
            pass
            
        except Exception as e:
            self.logger.error(f"Error getting historical predictions for {symbol}: {str(e)}")
            raise 

    async def get_next_day_prediction(self, symbol: str, model_type: str = "lstm") -> Dict[str, Any]:
        """Get next day prediction for a stock."""
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        if model_type not in ["lstm", "prophet"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            # Load the latest model and scaler
            model_result = await self.model_service.load_model(symbol, model_type)
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            # Get latest data
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            # Select only the required features
            df = data_result["data"]
            features = df[self.config.model.FEATURES].values
            
            # Scale the features
            scaled_features = scaler.transform(features)
            
            # Reshape for LSTM (samples, time steps, features)
            X = scaled_features.reshape(1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES))
            
            # Make prediction
            prediction = model.predict(X)
            
            # Inverse transform the prediction
            prediction = scaler.inverse_transform(
                np.hstack([prediction, np.zeros((prediction.shape[0], len(self.config.model.FEATURES) - 1))])
            )[:, 0]
            
            # Get confidence score
            confidence = self._calculate_confidence(prediction, scaled_features)
            
            # Prepare prediction details
            prediction_details = {
                "last_close": float(df["Close"].iloc[-1]),
                "predicted_change": float(prediction[0] - df["Close"].iloc[-1]),
                "predicted_change_percent": float((prediction[0] - df["Close"].iloc[-1]) / df["Close"].iloc[-1] * 100)
            }
            
            # Return result in legacy format
            return {
                "prediction": float(prediction[0]),
                "timestamp": datetime.now() + timedelta(days=1),
                "confidence_score": float(confidence),
                "model_version": model_result.get("version", "1.0.0"),
                "model_type": model_type,
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            self.logger.error(f"Error getting next day prediction for {symbol}: {str(e)}")
            raise

    async def get_historical_predictions(self, symbol: str, days: int = 7, model_type: str = "lstm") -> List[Dict[str, Any]]:
        """Get historical predictions for a stock."""
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        if model_type not in ["lstm", "prophet"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            # Load the latest model
            model_result = await self.model_service.load_model(symbol, model_type)
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load model for {symbol}")
            
            model = model_result["model"]
            
            # Get historical data
            data_result = await self.data_service.get_historical_data(symbol, days)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")
            
            predictions = []
            for i in range(len(data_result["data"])):
                # Prepare features
                features = calculate_technical_indicators(data_result["data"][:i+1])
                
                # Make prediction
                prediction = model.predict(features)
                
                # Get confidence score
                confidence = self._calculate_confidence(prediction, features)
                
                predictions.append({
                    "symbol": symbol,
                    "date": data_result["data"].iloc[i]["Date"].strftime("%Y-%m-%d"),
                    "predicted_price": float(prediction[0][0]),
                    "confidence": float(confidence),
                    "model_type": model_type,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting historical predictions for {symbol}: {str(e)}")
            raise

    def _calculate_confidence(self, prediction: np.ndarray, features: np.ndarray) -> float:
        """Calculate confidence score for a prediction."""
        # This is a simple implementation - you might want to use more sophisticated methods
        # like prediction intervals or model uncertainty estimates
        return 0.8  # Placeholder value 