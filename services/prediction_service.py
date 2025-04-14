"""
Stock prediction service.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random
from keras import backend as K  # Replace tensorflow with keras backend

from services.base_service import BaseService
from services.rabbitmq_service import RabbitMQService
from core.utils import (
    calculate_technical_indicators,
    validate_stock_symbol,
    format_prediction_response
)
from core.logging import logger
from services.model_service import ModelService
from services.data_service import DataService
from core.config import config

def configure_backend():
    """Configure the backend (PyTorch) for CPU-only operation."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    # PyTorch seed setting (equivalent to tf.random.set_seed)
    import torch
    torch.manual_seed(42)
    
    logger['prediction'].info("Backend configured for CPU-only operation with PyTorch")
    return False  # Always return False since we're using CPU-only version

# Configure backend at module level
USE_GPU = configure_backend()

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
        self.rabbitmq_service = RabbitMQService()
    
    async def initialize(self) -> None:
        """Initialize the prediction service."""
        try:
            await self._load_models()
            self._initialized = True
            self.logger.info("Prediction service initialized successfully with CPU configuration")
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction service: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.models.clear()
            self.scalers.clear()
            self._initialized = False
            self.rabbitmq_service.close()
            K.clear_session()  # Use Keras backend to clear session
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
        """Get stock price prediction for a given symbol."""
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        if model_type not in ["lstm", "prophet"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            if model_type == "prophet":
                # Check if Prophet model exists for this symbol
                prophet_models = [f.stem.split('_')[0] for f in self.model_service.prophet_dir.glob("*_prophet.json")]
                if symbol not in prophet_models:
                    available_models = ", ".join(sorted(prophet_models))
                    raise ValueError(
                        f"No Prophet model available for {symbol}. "
                        f"Available Prophet models: {available_models}"
                    )
            
            if model_type == "lstm":
                prediction = await self._get_lstm_prediction(symbol)
            else:
                prediction = await self._get_prophet_prediction(symbol)
            
            if prediction.get("status") == "error":
                return prediction
            
            # Publish prediction to RabbitMQ
            try:
                self.rabbitmq_service.publish_stock_quote(symbol, prediction)
            except Exception as e:
                self.logger.error(f"Failed to publish prediction to RabbitMQ: {str(e)}")
            
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
            
            model_result = await self.model_service.load_model(symbol, "lstm")
            if model_result["status"] != "success":
                self.logger.error(f"Model loading failed for {symbol}")
                raise RuntimeError(f"Failed to load LSTM model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                self.logger.error(f"Data fetch failed for {symbol}: {data_result.get('message', 'Unknown error')}")
                raise RuntimeError(f"Failed to get latest data for {symbol}: {data_result.get('message', 'Unknown error')}")
            
            df = data_result["data"]
            self.logger.info(f"Data shape for {symbol}: {df.shape}")
            self.logger.info(f"Data columns for {symbol}: {df.columns.tolist()}")
            
            # Validate data columns
            missing_features = [f for f in self.config.model.FEATURES if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {', '.join(missing_features)}")
            
            df = calculate_technical_indicators(df)
            self.logger.info(f"Data shape after technical indicators for {symbol}: {df.shape}")
            self.logger.info(f"Data columns after technical indicators for {symbol}: {df.columns.tolist()}")
            
            features_df = df[self.config.model.FEATURES].copy()
            self.logger.info(f"Features shape for {symbol}: {features_df.shape}")
            
            # Validate for NaN values
            if features_df.isna().any().any():
                nan_cols = features_df.columns[features_df.isna().any()].tolist()
                raise ValueError(f"Found NaN values in features: {', '.join(nan_cols)}")
            
            scaled_features = scaler.transform(features_df)
            self.logger.info(f"Scaled features shape for {symbol}: {scaled_features.shape}")
            
            # Ensure we have enough data for the sequence
            if len(scaled_features) < self.config.model.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Not enough data for prediction. Need {self.config.model.SEQUENCE_LENGTH} days, "
                    f"but only have {len(scaled_features)} days of data."
                )
            
            # Take the last SEQUENCE_LENGTH days of data
            sequence = scaled_features[-self.config.model.SEQUENCE_LENGTH:].reshape(
                1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES)
            )
            self.logger.info(f"Final sequence shape for {symbol}: {sequence.shape}")
            
            scaled_prediction = model.predict(sequence)
            
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
            
            change = prediction - last_close
            change_percent = (change / last_close) * 100
            
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
                scaled_change = scaled_prediction - last_scaled
                dummy = np.zeros((1, len(self.config.model.FEATURES)))
                dummy[0, close_idx] = scaled_change
                real_change = scaler.inverse_transform(dummy)[0, close_idx]
                prediction = last_close + real_change
                
                if np.isnan(prediction) or np.isinf(prediction):
                    self.logger.warning(f"Invalid prediction value detected: {prediction}. Using last close price.")
                    return float(last_close)
                
                relative_change = (prediction - last_close) / last_close
                if abs(relative_change) > 0.2:
                    conservative_price = last_close * (1 + (prediction - last_close) / last_close * 0.1)
                    self.logger.debug(f"Large change detected: {relative_change:.2%}. Using conservative estimate: {conservative_price:.2f}")
                    return float(conservative_price)
                
                return float(prediction)
                
            except Exception as e:
                self.logger.error(f"Error in inverse scaling: {str(e)}")
                return float(last_close)

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
            predictions = []
            for _ in range(num_samples):
                noisy_sequence = sequence + np.random.normal(0, 0.05, sequence.shape)
                pred = model.predict(noisy_sequence)
                predictions.append(pred[0, 0])
            
            predictions = np.array(predictions)
            dummy = np.zeros((len(predictions), len(self.config.model.FEATURES)))
            dummy[:, close_idx] = predictions
            unscaled_predictions = scaler.inverse_transform(dummy)[:, close_idx]
            
            mean_pred = np.mean(unscaled_predictions)
            std_pred = np.std(unscaled_predictions)
            relative_std = std_pred / abs(mean_pred) if mean_pred != 0 else float('inf')
            confidence = np.exp(-5 * relative_std)
            confidence = np.clip(confidence, 0, 1)
            
            self.logger.debug(f"Confidence calculation details:")
            self.logger.debug(f"Mean prediction: {mean_pred:.2f}")
            self.logger.debug(f"Std prediction: {std_pred:.2f}")
            self.logger.debug(f"Relative std: {relative_std:.4f}")
            self.logger.debug(f"Final confidence: {confidence:.4f}")
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating LSTM confidence: {str(e)}")
            return 0.5
    
    async def _get_prophet_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get prediction using Prophet model."""
        try:
            model_result = await self.model_service.load_model(symbol, "prophet")
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load Prophet model for {symbol}")
            
            model = model_result["model"]
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            df = data_result["data"]
            if 'RSI' not in df.columns:
                df = calculate_technical_indicators(df)
            
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Date']),
                'y': df['Close']
            })
            
            if hasattr(model, 'extra_regressors'):
                for regressor in model.extra_regressors.keys():
                    if regressor in df.columns:
                        prophet_df[regressor] = df[regressor]
                    else:
                        matching_cols = [col for col in df.columns if col.lower() == regressor.lower()]
                        if matching_cols:
                            prophet_df[regressor] = df[matching_cols[0]]
                        else:
                            raise ValueError(f"Regressor '{regressor}' missing from dataframe")
            
            future = model.make_future_dataframe(periods=1)
            if hasattr(model, 'extra_regressors'):
                for regressor in model.extra_regressors.keys():
                    if regressor in df.columns:
                        future[regressor] = df[regressor].iloc[-1]
                    else:
                        matching_cols = [col for col in df.columns if col.lower() == regressor.lower()]
                        if matching_cols:
                            future[regressor] = df[matching_cols[0]].iloc[-1]
                        else:
                            raise ValueError(f"Regressor '{regressor}' missing from dataframe")
            
            forecast = model.predict(future)
            prediction = forecast.iloc[-1]['yhat']
            last_price = df['Close'].iloc[-1]
            relative_change = (prediction - last_price) / last_price
            
            if abs(relative_change) > 0.2:
                conservative_price = last_price * (1 + (prediction - last_price) / last_price * 0.1)
                self.logger.debug(f"Large change detected: {relative_change:.2%}. Using conservative estimate: {conservative_price:.2f}")
                prediction = conservative_price
            
            confidence = self._calculate_prophet_confidence(forecast, last_price)
            
            if np.isnan(prediction) or np.isinf(prediction):
                self.logger.warning(f"Invalid prediction value detected: {prediction}. Using last close price.")
                prediction = last_price
            
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
            yhat = forecast.iloc[-1]['yhat']
            yhat_lower = forecast.iloc[-1]['yhat_lower']
            yhat_upper = forecast.iloc[-1]['yhat_upper']
            interval_width = yhat_upper - yhat_lower
            relative_width = interval_width / abs(yhat) if yhat != 0 else float('inf')
            confidence = np.exp(-relative_width)
            price_change = abs(yhat - last_price) / last_price
            if price_change > 0.1:
                confidence *= np.exp(-price_change + 0.1)
            confidence = float(np.clip(confidence, 0, 1))
            return confidence
        except Exception as e:
            self.logger.error(f"Error calculating Prophet confidence: {str(e)}")
            return 0.5
    
    async def get_historical_predictions(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get historical predictions for a given symbol."""
        try:
            model_result = await self.model_service.load_model(symbol, "lstm")
            if model_result["status"] != "success":
                raise RuntimeError(f"Failed to load LSTM model for {symbol}")
            
            model = model_result["model"]
            scaler = model_result["scaler"]
            
            data_result = await self.data_service.get_historical_data(symbol, days=days + self.config.model.SEQUENCE_LENGTH)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")
            
            df = data_result["data"]
            features_df = df[self.config.model.FEATURES].copy()
            scaled_features = scaler.transform(features_df)
            
            sequences = []
            for i in range(len(scaled_features) - self.config.model.SEQUENCE_LENGTH):
                sequence = scaled_features[i:i + self.config.model.SEQUENCE_LENGTH]
                sequences.append(sequence)
            
            predictions = []
            for sequence in sequences:
                sequence = sequence.reshape(1, self.config.model.SEQUENCE_LENGTH, len(self.config.model.FEATURES))
                scaled_prediction = model.predict(sequence)
                predictions.append(scaled_prediction[0, 0])
            
            last_close = df['Close'].iloc[-1]
            last_scaled = scaled_features[-1, self.config.model.FEATURES.index('Close')]
            predictions = [self._inverse_scale_lstm_prediction(
                pred,
                last_close,
                last_scaled,
                scaler,
                self.config.model.FEATURES.index('Close')
            ) for pred in predictions]
            
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