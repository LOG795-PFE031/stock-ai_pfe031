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
from core.config import config

def configure_tensorflow():
    """Configure TensorFlow for CPU-only operation."""
    # Configure TensorFlow for CPU-only
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Disable oneDNN optimization for half precision
    tf.config.experimental.enable_tensor_float_32_execution(False)
    
    # Set memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
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
        self.config = config
    
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
            
            # Check if prediction failed
            if prediction.get("status") == "error":
                return prediction
            
            # Format the response using the utility function
            return format_prediction_response(
                prediction=prediction["prediction"],
                confidence=prediction["confidence_score"],
                model_type=prediction["model_type"],
                model_version=prediction["model_version"],
                symbol=symbol
            )
            
        except Exception as e:
            self.logger.error(f"Error getting prediction for {symbol}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_lstm_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get prediction using LSTM model."""
        try:
            self.logger.info(f"Starting prediction for {symbol}")
            
            # Load the latest model and scaler
            model_result = await self.model_service.load_model(symbol, "lstm")
            if model_result["status"] != "success":
                self.logger.error(f"Model loading failed for {symbol}")
                raise RuntimeError(f"Failed to load LSTM model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            # Get latest data
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                self.logger.error(f"Data fetch failed for {symbol}")
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            # Prepare features and make prediction
            df = data_result["data"]
            df = calculate_technical_indicators(df)
            features_df = df[self.config.model.FEATURES].copy()
            
            # Scale features
            scaled_features = scaler.transform(features_df)
            sequence = scaled_features[-self.config.model.SEQUENCE_LENGTH:].reshape(
                1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES)
            )
            
            # Make prediction
            scaled_prediction = model.predict(sequence)
            
            # Process prediction
            last_close = df['Close'].iloc[-1]
            last_scaled = scaled_features[-1, self.config.model.FEATURES.index('Close')]
            
            prediction = self._inverse_scale_lstm_prediction(
                scaled_prediction[0, 0],
                last_close,
                last_scaled,
                scaler,
                self.config.model.FEATURES.index('Close')
            )
            
            confidence = self._calculate_lstm_confidence(
                sequence,
                scaled_prediction,
                model,
                scaler,
                self.config.model.FEATURES.index('Close')
            )
            
            # Calculate changes
            change = prediction - last_close
            change_percent = (change / last_close) * 100
            
            # Log prediction results
            self.logger.info(f"Prediction ready for {symbol}:")
            self.logger.info(f"Current: ${last_close:.2f} → Predicted: ${prediction:.2f}")
            self.logger.info(f"Change: ${change:+.2f} ({change_percent:+.2f}%)")
            self.logger.info(f"Confidence: {confidence:.1%}")
            
            return {
                "prediction": float(prediction),
                "timestamp": datetime.now() + timedelta(days=1),
                "confidence_score": float(confidence),
                "model_version": model_result.get("version", "1.0.0"),
                "model_type": "lstm",
                "prediction_details": {
                    "last_close": float(last_close),
                    "predicted_change": float(change),
                    "predicted_change_percent": float(change_percent),
                    "sequence_length": self.config.model.SEQUENCE_LENGTH,
                    "features_used": self.config.model.FEATURES
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": "lstm",
                "timestamp": datetime.now().isoformat()
            }

    def _inverse_scale_lstm_prediction(
        self,
        scaled_prediction: float,
        last_close: float,
        last_scaled: float,
        scaler: Any,
        close_idx: int
    ) -> float:
        """Inverse scale the LSTM prediction."""
        try:
            # Calculate the predicted change in scaled space
            scaled_change = scaled_prediction - last_scaled
            
            # Create a dummy array for inverse transform
            dummy = np.zeros((1, len(self.config.model.FEATURES)))
            dummy[0, close_idx] = scaled_change
            
            # Inverse transform the change
            real_change = scaler.inverse_transform(dummy)[0, close_idx]
            
            # Apply the change to the last actual price
            prediction = last_close + real_change
            
            # Handle invalid float values
            if np.isnan(prediction) or np.isinf(prediction):
                self.logger.warning(f"Invalid prediction value detected: {prediction}. Using last close price.")
                return float(last_close)
            
            # Calculate relative change
            relative_change = (prediction - last_close) / last_close
            
            # Apply conservative scaling for large changes (legacy approach)
            if abs(relative_change) > 0.2:  # 20% change threshold
                conservative_price = last_close * (1 + (prediction - last_close) / last_close * 0.1)
                self.logger.debug(f"Large change detected: {relative_change:.2%}. Using conservative estimate: {conservative_price:.2f}")
                return float(conservative_price)
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Error in inverse scaling: {str(e)}")
            return float(last_close)  # Return last close price as fallback

    def _calculate_lstm_confidence(
        self,
        sequence: np.ndarray,
        scaled_prediction: np.ndarray,
        model: Any,
        scaler: Any,
        close_idx: int,
        num_samples: int = 30
    ) -> float:
        """Calculate confidence score for LSTM prediction."""
        try:
            # Generate multiple predictions with slight variations
            predictions = []
            for _ in range(num_samples):
                # Add random noise to the sequence (increased noise for better variance)
                noisy_sequence = sequence + np.random.normal(0, 0.05, sequence.shape)
                # Make prediction
                pred = model.predict(noisy_sequence)
                predictions.append(pred[0, 0])
            
            predictions = np.array(predictions)
            
            # Calculate statistics in the original price space
            dummy = np.zeros((len(predictions), len(self.config.model.FEATURES)))
            dummy[:, close_idx] = predictions
            unscaled_predictions = scaler.inverse_transform(dummy)[:, close_idx]
            
            # Calculate relative standard deviation (coefficient of variation)
            mean_pred = np.mean(unscaled_predictions)
            std_pred = np.std(unscaled_predictions)
            relative_std = std_pred / abs(mean_pred) if mean_pred != 0 else float('inf')
            
            # Convert to confidence score
            # Use exponential decay: confidence = exp(-k * relative_std)
            # k = 5 means:
            # - 1% relative std -> 0.95 confidence
            # - 5% relative std -> 0.78 confidence
            # - 10% relative std -> 0.61 confidence
            # - 20% relative std -> 0.37 confidence
            confidence = np.exp(-5 * relative_std)
            
            # Clip confidence to [0, 1] range
            confidence = np.clip(confidence, 0, 1)
            
            # Log the confidence calculation details for debugging
            self.logger.debug(f"Confidence calculation details:")
            self.logger.debug(f"Mean prediction: {mean_pred:.2f}")
            self.logger.debug(f"Std prediction: {std_pred:.2f}")
            self.logger.debug(f"Relative std: {relative_std:.4f}")
            self.logger.debug(f"Final confidence: {confidence:.4f}")
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating LSTM confidence: {str(e)}")
            return 0.5  # Return neutral confidence on error
    
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
            self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            
            # Calculate technical indicators if not already present
            if 'RSI' not in df.columns:
                df = calculate_technical_indicators(df)
                self.logger.debug(f"Added technical indicators. New columns: {df.columns.tolist()}")
            
            # Create Prophet dataframe with all required columns
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Date']),
                'y': df['Close']
            })
            self.logger.debug(f"Created Prophet DataFrame with columns: {prophet_df.columns.tolist()}")
            
            # Add regressors if they exist in the model
            if hasattr(model, 'extra_regressors'):
                self.logger.debug(f"Model has {len(model.extra_regressors)} regressors")
                for regressor in model.extra_regressors.keys():
                    # Try exact match first
                    if regressor in df.columns:
                        prophet_df[regressor] = df[regressor]
                        self.logger.debug(f"Added regressor {regressor} from exact column match")
                    else:
                        # Try case-insensitive match
                        matching_cols = [col for col in df.columns if col.lower() == regressor.lower()]
                        if matching_cols:
                            prophet_df[regressor] = df[matching_cols[0]]
                            self.logger.debug(f"Added regressor {regressor} from case-insensitive match with {matching_cols[0]}")
                        else:
                            self.logger.warning(f"Regressor {regressor} not found in DataFrame columns: {df.columns.tolist()}")
                            raise ValueError(f"Regressor '{regressor}' missing from dataframe")
            
            self.logger.debug(f"Final Prophet DataFrame columns: {prophet_df.columns.tolist()}")
            
            # Make future dataframe with all required columns
            future = model.make_future_dataframe(periods=1)
            self.logger.debug(f"Created future DataFrame with columns: {future.columns.tolist()}")
            
            # Add regressors to future dataframe
            if hasattr(model, 'extra_regressors'):
                self.logger.debug("Adding regressors to future DataFrame")
                for regressor in model.extra_regressors.keys():
                    # Try exact match first
                    if regressor in df.columns:
                        future[regressor] = df[regressor].iloc[-1]
                        self.logger.debug(f"Added regressor {regressor} from exact column match")
                    else:
                        # Try case-insensitive match
                        matching_cols = [col for col in df.columns if col.lower() == regressor.lower()]
                        if matching_cols:
                            future[regressor] = df[matching_cols[0]].iloc[-1]
                            self.logger.debug(f"Added regressor {regressor} from case-insensitive match with {matching_cols[0]}")
                        else:
                            self.logger.warning(f"Regressor {regressor} not found in DataFrame columns: {df.columns.tolist()}")
                            raise ValueError(f"Regressor '{regressor}' missing from dataframe")
            
            self.logger.debug(f"Final future DataFrame columns: {future.columns.tolist()}")
            
            # Make prediction
            forecast = model.predict(future)
            prediction = forecast.iloc[-1]['yhat']
            
            # Get the last actual close price
            last_price = df['Close'].iloc[-1]
            
            # Calculate relative change
            relative_change = (prediction - last_price) / last_price
            
            # Apply conservative scaling for large changes (legacy approach)
            if abs(relative_change) > 0.2:  # 20% change threshold
                conservative_price = last_price * (1 + (prediction - last_price) / last_price * 0.1)
                self.logger.debug(f"Large change detected: {relative_change:.2%}. Using conservative estimate: {conservative_price:.2f}")
                prediction = conservative_price
            
            # Calculate confidence using the new method
            confidence = self._calculate_prophet_confidence(forecast, last_price)
            
            # Handle invalid prediction values
            if np.isnan(prediction) or np.isinf(prediction):
                self.logger.warning(f"Invalid prediction value detected: {prediction}. Using last close price.")
                prediction = last_price
            
            # Prepare prediction details
            prediction_details = {
                "last_close": float(last_price),
                "predicted_change": float(prediction - last_price),
                "predicted_change_percent": float((prediction - last_price) / last_price * 100),
                "confidence_interval": {
                    "lower": float(forecast.iloc[-1]['yhat_lower']),
                    "upper": float(forecast.iloc[-1]['yhat_upper'])
                },
                "status": "within_normal_range" if abs(relative_change) <= 0.2 else "large_change_detected"
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
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": "prophet",
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_prophet_confidence(
        self,
        forecast: pd.DataFrame,
        last_price: float
    ) -> float:
        """Calculate confidence score for Prophet prediction."""
        try:
            # Get prediction interval
            yhat = forecast.iloc[-1]['yhat']
            yhat_lower = forecast.iloc[-1]['yhat_lower']
            yhat_upper = forecast.iloc[-1]['yhat_upper']
            
            # Calculate interval width relative to prediction
            interval_width = yhat_upper - yhat_lower
            relative_width = interval_width / abs(yhat) if yhat != 0 else float('inf')
            
            # Convert to confidence score (inverse relationship with relative width)
            # Wider intervals mean lower confidence
            confidence = np.exp(-relative_width)
            
            # Adjust confidence based on prediction deviation from last price
            price_change = abs(yhat - last_price) / last_price
            if price_change > 0.1:  # More than 10% change
                confidence *= np.exp(-price_change + 0.1)
            
            # Ensure confidence is between 0 and 1
            confidence = float(np.clip(confidence, 0, 1))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating Prophet confidence: {str(e)}")
            return 0.5  # Return moderate confidence as fallback
    
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
            # Load the latest model and scaler
            model_result = await self.model_service.load_model(symbol, "lstm")
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load LSTM model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            # Get historical data
            data_result = await self.data_service.get_historical_data(symbol, days=days + self.config.model.SEQUENCE_LENGTH)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")
            
            # Prepare features
            df = data_result["data"]
            features_df = df[self.config.model.FEATURES].copy()
            features_df.columns = self.config.model.FEATURES
            
            # Scale features
            scaled_features = scaler.transform(features_df)
            
            # Create sequences for LSTM
            sequences = []
            for i in range(len(scaled_features) - self.config.model.SEQUENCE_LENGTH):
                sequence = scaled_features[i:i + self.config.model.SEQUENCE_LENGTH]
                sequences.append(sequence)
            
            # Make predictions
            predictions = []
            for sequence in sequences:
                # Reshape sequence for prediction
                sequence = sequence.reshape(1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES))
                # Make prediction
                scaled_prediction = model.predict(sequence)
                predictions.append(scaled_prediction[0, 0])
            
            # Inverse transform predictions
            last_close = df['Close'].iloc[-1]
            last_scaled = scaled_features[-1, self.config.model.FEATURES.index('Close')]
            predictions = [self._inverse_scale_lstm_prediction(
                pred,
                last_close,
                last_scaled,
                scaler,
                self.config.model.FEATURES.index('Close')
            ) for pred in predictions]
            
            # Prepare response
            historical_predictions = []
            for i, pred in enumerate(predictions):
                historical_predictions.append({
                    "date": df['Date'].iloc[i + self.config.model.SEQUENCE_LENGTH].isoformat(),
                    "prediction": float(pred),
                    "actual": float(df['Close'].iloc[i + self.config.model.SEQUENCE_LENGTH]),
                    "error": float(pred - df['Close'].iloc[i + self.config.model.SEQUENCE_LENGTH])
                })
            
            return {
                "status": "success",
                "symbol": symbol,
                "historical_predictions": historical_predictions,
                "model_version": model_result.get("version", "1.0.0"),
                "model_type": "lstm"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical predictions for {symbol}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "historical_predictions": [],
                "model_version": "unknown",
                "model_type": "lstm"
            }

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
            scaler = model_result["scaler"]
            
            # Get historical data
            data_result = await self.data_service.get_historical_data(symbol, days)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")
            
            predictions = []
            for i in range(len(data_result["data"])):
                # Prepare features
                features = calculate_technical_indicators(data_result["data"][:i+1])
                features = features[self.config.model.FEATURES].values
                features = scaler.transform(features)
                features = features.reshape(1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES))
                
                # Make prediction
                prediction = model.predict(features)
                
                # Inverse transform the prediction
                prediction = scaler.inverse_transform(
                    np.hstack([prediction, np.zeros((prediction.shape[0], len(self.config.model.FEATURES) - 1))])
                )[:, 0]
                
                # Get confidence score
                confidence = self._calculate_confidence(prediction, features)
                
                predictions.append({
                    "symbol": symbol,
                    "date": data_result["data"].iloc[i]["Date"].strftime("%Y-%m-%d"),
                    "predicted_price": float(prediction[0]),
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