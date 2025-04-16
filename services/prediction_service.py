"""
Stock prediction service.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import random
import asyncio
from keras import backend as K  # Replace tensorflow with keras backend
import joblib
import json
from pathlib import Path

from services.base_service import BaseService
from services.rabbitmq_service import RabbitMQService
from services.training_service import TrainingService
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
        self._publish_task = None
        self._stop_publishing = False
        self.training_service = TrainingService(model_service, data_service)
    
    async def initialize(self) -> None:
        """Initialize the prediction service."""
        try:
            # Initialize services
            await self.data_service.initialize()
            await self.model_service.initialize()
            await self.training_service.initialize()
            
            # Train missing Prophet models
            await self._train_missing_prophet_models()
            
            # Load models
            await self._load_models()
            
            self._initialized = True
            self.logger.info("Prediction service initialized successfully with CPU configuration")
            
            # Set up day-started callback
            self.rabbitmq_service.set_day_started_callback(self._on_day_started)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction service: {str(e)}")
            raise
    
    async def _train_missing_prophet_models(self) -> None:
        """Train Prophet models for symbols that don't have them."""
        try:
            # List of symbols that should have Prophet models
            target_symbols = [
                "AAPL", "ADBE", "AMZN", "CSCO", "GOOGL",
                "INTC", "META", "MSFT", "NVDA", "TSLA"
            ]
            
            # Get existing Prophet models
            existing_models = [f.stem.split('_')[0] for f in Path("models/prophet").glob("*_prophet.joblib")]
            
            # Find missing models
            missing_models = [symbol for symbol in target_symbols if symbol not in existing_models]
            
            if not missing_models:
                self.logger.info("All Prophet models are up to date")
                return
            
            self.logger.info(f"Training Prophet models for: {', '.join(missing_models)}")
            
            # Train missing models
            for symbol in missing_models:
                try:
                    self.logger.info(f"Training Prophet model for {symbol}")
                    
                    result = await self.training_service.train_model(
                        symbol=symbol,
                        model_type="prophet",
                        start_date=datetime.now() - timedelta(days=365*2),  # 2 years of data
                        end_date=datetime.now(),
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0,
                        holidays_prior_scale=10.0,
                        seasonality_mode='multiplicative'
                    )
                    
                    if result.get("status") == "success":
                        self.logger.info(f"Successfully trained Prophet model for {symbol}")
                        self.logger.info(f"Model metrics: {result.get('metrics', {})}")
                    else:
                        self.logger.error(f"Failed to train Prophet model for {symbol}: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    self.logger.error(f"Error training Prophet model for {symbol}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in _train_missing_prophet_models: {str(e)}")
            raise
    
    def _on_day_started(self, message: Dict[str, Any]) -> None:
        """Handle day-started event."""
        try:
            self.logger.info("Day started event received, publishing predictions")
            # Create a task to run the coroutine in the background
            asyncio.create_task(self._publish_all_predictions())
        except Exception as e:
            self.logger.error(f"Error handling day-started event: {str(e)}")
    
    async def _publish_all_predictions(self) -> None:
        """Publish predictions for all available models."""
        try:
            # Get all available models
            lstm_models = list(self.model_service._specific_models.keys())
            prophet_models = [f.stem.split('_')[0] for f in self.model_service.prophet_dir.glob("*_prophet.joblib")]
            
            # Combine and deduplicate symbols
            all_symbols = list(set(lstm_models + prophet_models))
            
            self.logger.info(f"Publishing predictions for {len(all_symbols)} symbols")
            
            # Create tasks for all predictions
            tasks = []
            for symbol in all_symbols:
                # Try LSTM first
                if symbol in lstm_models:
                    tasks.append(self.get_prediction(symbol, "lstm"))
                # Then try Prophet if available
                if symbol in prophet_models:
                    tasks.append(self.get_prediction(symbol, "prophet"))
            
            # Process predictions as they complete
            success_count = 0
            error_count = 0
            successful_predictions = []
            failed_predictions = []
            
            # Create a progress tracking task
            total_tasks = len(tasks)
            completed_tasks = 0
            
            # Process results as they complete
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    completed_tasks += 1
                    
                    if isinstance(result, dict) and result.get("status") != "error":
                        success_count += 1
                        pred_info = {
                            "symbol": result.get("symbol", "unknown"),
                            "model_type": result.get("model_type", "unknown"),
                            "prediction": result.get("prediction", 0.0),
                            "confidence": result.get("confidence_score", 0.0)
                        }
                        successful_predictions.append(pred_info)
                        self.logger.info(
                            f"✅ [{completed_tasks}/{total_tasks}] {pred_info['symbol']} ({pred_info['model_type'].upper()}): "
                            f"${pred_info['prediction']:.2f} (Confidence: {pred_info['confidence']:.1%})"
                        )
                    else:
                        error_count += 1
                        fail_info = {
                            "symbol": result.get("symbol", "unknown") if isinstance(result, dict) else "unknown",
                            "model_type": result.get("model_type", "unknown") if isinstance(result, dict) else "unknown",
                            "error": str(result) if not isinstance(result, dict) else result.get("error", "unknown error")
                        }
                        failed_predictions.append(fail_info)
                        self.logger.warning(
                            f"❌ [{completed_tasks}/{total_tasks}] {fail_info['symbol']} ({fail_info['model_type'].upper()}): {fail_info['error']}"
                        )
                except Exception as e:
                    error_count += 1
                    self.logger.error(f"❌ [{completed_tasks}/{total_tasks}] Error processing prediction: {str(e)}")
            
            # Log final summary
            self.logger.info(f"📊 Prediction Summary:")
            self.logger.info(f"   Total predictions attempted: {total_tasks}")
            self.logger.info(f"   Successful predictions: {success_count}")
            self.logger.info(f"   Failed predictions: {error_count}")
            
            # Log successful predictions summary
            if successful_predictions:
                self.logger.info("✨ Successful Predictions Summary:")
                for pred in successful_predictions:
                    self.logger.info(
                        f"   {pred['symbol']} ({pred['model_type'].upper()}): "
                        f"${pred['prediction']:.2f} (Confidence: {pred['confidence']:.1%})"
                    )
            
            # Log failed predictions summary
            if failed_predictions:
                self.logger.warning("❌ Failed Predictions Summary:")
                for fail in failed_predictions:
                    self.logger.warning(f"   {fail['symbol']} ({fail['model_type'].upper()}): {fail['error']}")
            
        except Exception as e:
            self.logger.error(f"Error publishing predictions: {str(e)}")
    
    async def start_auto_publishing(self, interval_minutes: int = 5) -> None:
        """
        Start automatically publishing predictions for all available models.
        
        Args:
            interval_minutes: Time interval between publishing predictions
        """
        if not self._initialized:
            raise RuntimeError("Prediction service not initialized")
        
        self._stop_publishing = False
        self._publish_task = asyncio.create_task(self._publish_loop())
        self.logger.info(f"Started auto-publishing predictions every {interval_minutes} minutes")
    
    async def stop_auto_publishing(self) -> None:
        """Stop the auto-publishing task."""
        if self._publish_task:
            self._stop_publishing = True
            await self._publish_task
            self._publish_task = None
            self.logger.info("Stopped auto-publishing predictions")
    
    async def _publish_loop(self):
        """Main loop for publishing predictions."""
        self.logger.info("✨ Starting prediction publishing loop")
        
        while not self._stop_publishing:
            try:
                # Get all symbols that need predictions
                symbols = self._get_symbols_for_prediction()
                if not symbols:
                    self.logger.info("✨ No symbols need predictions, waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                self.logger.info(f"✨ Found {len(symbols)} symbols needing predictions")
                
                # Create tasks for all predictions
                tasks = []
                for symbol in symbols:
                    # Create tasks for both LSTM and Prophet predictions
                    lstm_task = asyncio.create_task(self._get_lstm_prediction(symbol))
                    tasks.append(lstm_task)
                    
                    # Only add Prophet task if model exists
                    prophet_models = [f.stem.split('_')[0] for f in self.model_service.prophet_dir.glob("*_prophet.joblib")]
                    if symbol in prophet_models:
                        prophet_task = asyncio.create_task(self._get_prophet_prediction(symbol))
                        tasks.append(prophet_task)
                
                # Process predictions as they complete
                total_tasks = len(tasks)
                completed_tasks = 0
                successful_predictions = []
                failed_predictions = []
                
                for completed_task in asyncio.as_completed(tasks):
                    try:
                        result = await completed_task
                        completed_tasks += 1
                        
                        if result:
                            symbol, prediction, model_type = result
                            # Publish the prediction
                            if await self._publish_prediction(symbol, prediction, model_type):
                                successful_predictions.append((symbol, prediction, model_type))
                                self.logger.info(
                                    f"✨ ✅ [{completed_tasks}/{total_tasks}] {symbol} ({model_type}): "
                                    f"${prediction['prediction']:.2f} (Confidence: {prediction['confidence_score']:.1%})"
                                )
                            else:
                                failed_predictions.append((symbol, model_type))
                                self.logger.error(
                                    f"✨ ❌ [{completed_tasks}/{total_tasks}] Failed to publish prediction for {symbol} ({model_type})"
                                )
                    except Exception as e:
                        completed_tasks += 1
                        self.logger.error(f"❌ Error processing prediction task: {str(e)}")
                        failed_predictions.append(("unknown", "unknown"))
                
                # Log final summary
                self.logger.info("✨ 📊 Prediction Summary:")
                self.logger.info(f"✨    Total predictions attempted: {total_tasks}")
                self.logger.info(f"✨    Successful predictions: {len(successful_predictions)}")
                self.logger.info(f"✨    Failed predictions: {len(failed_predictions)}")
                
                if successful_predictions:
                    self.logger.info("✨ Successful Predictions Summary:")
                    for symbol, prediction, model_type in successful_predictions:
                        self.logger.info(
                            f"✨    {symbol} ({model_type}): "
                            f"${prediction['prediction']:.2f} (Confidence: {prediction['confidence_score']:.1%})"
                        )
                
                if failed_predictions:
                    self.logger.info("✨ Failed Predictions Summary:")
                    for symbol, model_type in failed_predictions:
                        self.logger.info(f"✨    {symbol} ({model_type})")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Wait 1 minute before next batch
                
            except Exception as e:
                self.logger.error(f"❌ Error in publish loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.stop_auto_publishing()
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
                prophet_models = [f.stem.split('_')[0] for f in self.model_service.prophet_dir.glob("*_prophet.joblib")]
                if symbol not in prophet_models:
                    available_models = ", ".join(sorted(prophet_models))
                    self.logger.warning(f"No Prophet model available for {symbol}. Available Prophet models: {available_models}")
                    return {
                        "status": "error",
                        "error": f"No Prophet model available for {symbol}",
                        "symbol": symbol,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat()
                    }
            
            if model_type == "lstm":
                prediction = await self._get_lstm_prediction(symbol)
            else:
                prediction = await self._get_prophet_prediction(symbol)
            
            if prediction.get("status") == "error":
                return prediction
            
            # Format prediction for RabbitMQ
            rabbitmq_prediction = {
                "prediction": prediction["prediction"],
                "confidence_score": prediction["confidence_score"],
                "model_type": prediction["model_type"],
                "model_version": prediction["model_version"],
                "timestamp": prediction["timestamp"].isoformat() if isinstance(prediction["timestamp"], datetime) else prediction["timestamp"]
            }
            
            # Publish prediction to RabbitMQ
            try:
                self.rabbitmq_service.publish_stock_quote(symbol, rabbitmq_prediction)
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
    
    async def _get_lstm_prediction(self, symbol: str) -> Tuple[str, Dict[str, Any], str]:
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
            
            prediction_data = {
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
            
            return symbol, prediction_data, "lstm"
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {str(e)}")
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
        close_idx: int
    ) -> float:
        """Calculate confidence score for LSTM prediction."""
        try:
            # Calculate historical volatility from the sequence
            historical_volatility = np.std(sequence[:, :, close_idx]) / np.mean(sequence[:, :, close_idx])
            
            # Calculate prediction magnitude (relative to last price)
            last_price = sequence[-1, -1, close_idx]
            prediction_magnitude = abs(scaled_prediction[0, 0] - last_price) / abs(last_price) if last_price != 0 else float('inf')
            
            # Calculate confidence based on volatility and prediction magnitude
            # Lower volatility and smaller prediction changes result in higher confidence
            volatility_factor = np.exp(-2 * historical_volatility)
            magnitude_factor = np.exp(-3 * prediction_magnitude)
            
            # Combine factors with weights
            confidence = 0.7 * volatility_factor + 0.3 * magnitude_factor
            confidence = np.clip(confidence, 0, 1)
            
            self.logger.debug(f"Confidence calculation details:")
            self.logger.debug(f"Historical volatility: {historical_volatility:.4f}")
            self.logger.debug(f"Prediction magnitude: {prediction_magnitude:.4f}")
            self.logger.debug(f"Volatility factor: {volatility_factor:.4f}")
            self.logger.debug(f"Magnitude factor: {magnitude_factor:.4f}")
            self.logger.debug(f"Final confidence: {confidence:.4f}")
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating LSTM confidence: {str(e)}")
            return 0.5
    
    async def _get_prophet_prediction(self, symbol: str) -> Tuple[str, Dict[str, Any], str]:
        """Get prediction using Prophet model."""
        try:
            # Load model from joblib file
            prophet_dir = Path("models/prophet")
            model_file = prophet_dir / f"{symbol}_prophet.joblib"
            metadata_file = prophet_dir / f"{symbol}_prophet_metadata.json"
            
            if not model_file.exists():
                raise RuntimeError(f"Failed to load Prophet model for {symbol}: Model file not found")
            
            # Load the model and metadata
            model = joblib.load(model_file)
            with open(metadata_file, "r") as f:
                model_metadata = json.load(f)
            
            # Get latest data
            data_result = await self.data_service.get_latest_data(symbol)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get latest data for {symbol}")
            
            df = data_result["data"]
            if 'RSI' not in df.columns:
                df = calculate_technical_indicators(df)
            
            # Ensure dates are timezone-aware UTC
            dates = pd.to_datetime(df['Date'])
            if dates.dt.tz is None:
                dates = dates.dt.tz_localize('UTC')
            else:
                dates = dates.dt.tz_convert('UTC')
            
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': df['Close']
            })
            
            # Add regressors to the dataframe
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
            
            # Create future dataframe with timezone-aware dates
            future = model.make_future_dataframe(periods=1)
            future_dates = pd.to_datetime(future['ds'])
            if future_dates.dt.tz is None:
                future_dates = future_dates.dt.tz_localize('UTC')
            else:
                future_dates = future_dates.dt.tz_convert('UTC')
            future['ds'] = future_dates
            
            # Add regressors to future dataframe
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
            
            # Make prediction
            forecast = model.predict(future)
            
            # Get the last prediction
            last_prediction = forecast.iloc[-1]
            
            # Calculate confidence based on prediction interval width
            confidence = 100 * (1 - (last_prediction['yhat_upper'] - last_prediction['yhat_lower']) / last_prediction['yhat'])
            
            prediction_data = {
                "prediction": float(last_prediction['yhat']),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence_score": float(confidence),
                "model_version": model_metadata.get("model_version", "1.0.0"),
                "model_type": "prophet",
                "prediction_details": {
                    "yhat_lower": float(last_prediction['yhat_lower']),
                    "yhat_upper": float(last_prediction['yhat_upper']),
                    "interval_width": float(last_prediction['yhat_upper'] - last_prediction['yhat_lower'])
                }
            }
            
            return symbol, prediction_data, "prophet"
            
        except Exception as e:
            self.logger.error(f"Error getting Prophet prediction for {symbol}: {str(e)}")
            raise

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

    def _get_symbols_for_prediction(self) -> List[str]:
        """Get list of symbols that need predictions."""
        try:
            # Get all available models
            lstm_models = list(self.model_service._specific_models.keys())
            prophet_models = [f.stem.split('_')[0] for f in self.model_service.prophet_dir.glob("*_prophet.joblib")]
            
            # Combine and deduplicate symbols
            all_symbols = list(set(lstm_models + prophet_models))
            
            if not all_symbols:
                self.logger.warning("⚠️ No models available for prediction")
                return []
            
            self.logger.info(f"✨ Found {len(all_symbols)} symbols with available models")
            return all_symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols for prediction: {str(e)}")
            return []

    async def _publish_prediction(self, symbol: str, prediction: Dict[str, Any], model_type: str) -> bool:
        """
        Publish a prediction to RabbitMQ.
        
        Args:
            symbol: Stock symbol
            prediction: Prediction data
            model_type: Type of model used (lstm or prophet)
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        try:
            # Format prediction for RabbitMQ
            rabbitmq_prediction = {
                "prediction": prediction["prediction"],
                "confidence_score": prediction["confidence_score"],
                "model_type": model_type,
                "model_version": prediction.get("model_version", "1.0.0"),
                "timestamp": prediction.get("timestamp", datetime.now().isoformat())
            }
            
            # Ensure RabbitMQ connection is ready
            if not self.rabbitmq_service._event.is_set():
                self.logger.info("✨ Waiting for RabbitMQ connection to be ready...")
                if not self.rabbitmq_service._event.wait(timeout=30):
                    self.logger.error("❌ Timeout waiting for RabbitMQ connection to be ready")
                    return False
            
            # Publish to RabbitMQ
            success = self.rabbitmq_service.publish_stock_quote(symbol, rabbitmq_prediction)
            
            if success:
                self.logger.info(
                    f"✅ Published prediction for {symbol} ({model_type}): "
                    f"${prediction['prediction']:.2f} (Confidence: {prediction['confidence_score']:.1%})"
                )
            else:
                self.logger.error(f"❌ Failed to publish prediction for {symbol} ({model_type})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error publishing prediction for {symbol}: {str(e)}")
            return False