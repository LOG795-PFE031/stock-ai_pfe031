"""
Service for model storage and versioning.
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import shutil
from pathlib import Path
import tensorflow as tf
import joblib
from packaging import version
import pandas as pd

from services.base_service import BaseService
from core.config import config
from core.logging import logger

class ModelService(BaseService):
    """Service for managing model storage and versioning."""
    
    def __init__(self):
        super().__init__()
        self.model_dir = Path("/app/models/specific").resolve()
        self.prophet_dir = Path("/app/models/prophet").resolve()
        self._model_metadata = {}
        self.model_version = "1.0.0"
        self.logger = logger['model']
        self.logger.info(f"Model service initialized with model_dir: {self.model_dir}")
        self.logger.info(f"Model service initialized with prophet_dir: {self.prophet_dir}")
        self._specific_models = {}
        self._specific_scalers = {}
        self._general_model = None
        self._general_scalers = None
    
    async def initialize(self) -> None:
        """Initialize the model service."""
        try:
            await self._load_model_metadata()
            await self._load_models()
            self._initialized = True
            self.logger.info("Model service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize model service: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources used by the model service."""
        try:
            await self._save_model_metadata()
            self._initialized = False
            self.logger.info("Model service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to clean up model service: {str(e)}")
            raise
    
    async def save_model(
        self,
        symbol: str,
        model_type: str,
        model: Any,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save a trained model with its metrics and metadata."""
        try:
            if not self._initialized:
                raise RuntimeError("Model service not initialized")
            
            # Create model directory
            model_dir = self._get_version_dir(symbol, model_type)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / f"{symbol}_model.keras"
            if isinstance(model, tf.keras.Model):
                tf.saved_model.save(model, str(model_path))
            else:
                model_path = model_dir / f"{symbol}_{model_type}_model.joblib"
                joblib.dump(model, model_path)
            
            # Save metrics
            metrics_path = model_dir / f"{symbol}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            
            # Save metadata
            if metadata:
                metadata_path = model_dir / f"{symbol}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
            
            # Update model metadata
            model_key = f"{symbol}_{model_type}"
            if model_key not in self._model_metadata:
                self._model_metadata[model_key] = []
            
            self._model_metadata[model_key].append({
                "version": self.model_version,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "metadata": metadata or {}
            })
            
            await self._save_model_metadata()
            
            return {
                "status": "success",
                "message": f"Model saved successfully for {symbol} ({model_type})",
                "version": self.model_version,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return self.format_error_response(e)
    
    async def _load_models(self) -> None:
        """Load all models from disk."""
        try:
            # Load LSTM models
            for symbol_dir in self.model_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    try:
                        # Load scaler
                        scaler_path = symbol_dir / f"{symbol}_scaler.gz"
                        if scaler_path.exists():
                            self._specific_scalers[symbol] = joblib.load(scaler_path)
                            self.logger.info(f"Loaded scaler for {symbol}")
                        else:
                            self.logger.warning(f"Scaler file not found for {symbol} at {scaler_path}")
                        
                        # Load model
                        model_path = symbol_dir / f"{symbol}_model.keras"
                        if model_path.exists():
                            self._specific_models[symbol] = tf.keras.models.load_model(str(model_path))
                            self.logger.info(f"Loaded LSTM model for {symbol}")
                        else:
                            self.logger.warning(f"Model file not found for {symbol} at {model_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading model for {symbol}: {str(e)}")
            
            # Load Prophet models
            prophet_dir = self.model_dir / "prophet"
            if prophet_dir.exists():
                for symbol_dir in prophet_dir.iterdir():
                    if symbol_dir.is_dir():
                        symbol = symbol_dir.name
                        try:
                            model_path = symbol_dir / f"{symbol}_model.pkl"
                            if model_path.exists():
                                self._specific_models[symbol] = joblib.load(model_path)
                                self.logger.info(f"Loaded Prophet model for {symbol}")
                            else:
                                self.logger.warning(f"Prophet model file not found for {symbol} at {model_path}")
                        except Exception as e:
                            self.logger.error(f"Error loading Prophet model for {symbol}: {str(e)}")
            else:
                self.logger.warning(f"Prophet model directory not found at {prophet_dir}")
            
            self.logger.info("Model loading completed")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    async def load_model(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Load a trained model and its scaler."""
        try:
            self.logger.info(f"Attempting to load {model_type} model for {symbol}")
            
            # Get the correct directory for the model type
            if model_type == "lstm":
                model_dir = self.model_dir / symbol
                model_path = model_dir / f"{symbol}_model.keras"
                scaler_path = model_dir / f"{symbol}_scaler.gz"
                self.logger.info(f"LSTM model directory: {model_dir}")
                self.logger.info(f"LSTM model path: {model_path}")
                self.logger.info(f"LSTM scaler path: {scaler_path}")
            elif model_type == "prophet":
                model_dir = self.prophet_dir
                model_path = model_dir / f"{symbol}_prophet.json"
                scaler_path = None
                self.logger.info(f"Prophet model directory: {model_dir}")
                self.logger.info(f"Prophet model path: {model_path}")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Check if model exists
            if not model_path.exists():
                self.logger.error(f"Model file not found at path: {model_path}")
                self.logger.info(f"Directory contents: {list(model_dir.glob('*'))}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load model
            if model_type == "prophet":
                self.logger.info("Loading Prophet model from JSON")
                from prophet import Prophet
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                
                # Create a new Prophet model with the saved parameters
                model = Prophet(
                    changepoint_prior_scale=model_data['params']['changepoint_prior_scale'],
                    seasonality_prior_scale=model_data['params']['seasonality_prior_scale'],
                    seasonality_mode=model_data['params']['seasonality_mode'],
                    daily_seasonality=model_data['params']['daily_seasonality'],
                    weekly_seasonality=model_data['params']['weekly_seasonality'],
                    yearly_seasonality=model_data['params']['yearly_seasonality']
                )
                
                # Add regressors if they exist
                if 'regressors' in model_data and model_data['regressors']:
                    for regressor in model_data['regressors']:
                        model.add_regressor(regressor)
                
                # Add the training data if it exists
                if 'last_data' in model_data and model_data['last_data']:
                    df = pd.DataFrame(model_data['last_data'])
                    df['ds'] = pd.to_datetime(df['Date'])
                    df['y'] = df['Close']
                    model.fit(df)
            else:
                self.logger.info("Loading LSTM model")
                # Try loading as a Keras model first
                try:
                    self.logger.info("Attempting to load as Keras model")
                    model = tf.keras.models.load_model(str(model_path), compile=False)
                    self.logger.info("Successfully loaded as Keras model")
                except Exception as e:
                    self.logger.warning(f"Failed to load as Keras model: {str(e)}")
                    # If that fails, try loading as SavedModel
                    try:
                        self.logger.info("Attempting to load as SavedModel")
                        model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects={})
                        self.logger.info("Successfully loaded as SavedModel")
                    except Exception as e:
                        self.logger.error(f"Failed to load model in any format: {str(e)}")
                        raise
            
            # Load scaler if it's an LSTM model
            scaler = None
            if model_type == "lstm":
                if scaler_path.exists():
                    self.logger.info("Loading LSTM scaler")
                    scaler = joblib.load(str(scaler_path))
                    self.logger.info("Successfully loaded LSTM scaler")
                else:
                    self.logger.error(f"Scaler file not found at path: {scaler_path}")
                    raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            self.logger.info(f"Successfully loaded {model_type} model for {symbol}")
            return {
                "status": "success",
                "model": model,
                "scaler": scaler,
                "version": self.model_version
            }
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            }
    
    async def list_models(
        self,
        symbol: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all available models with their metadata."""
        try:
            if not self._initialized:
                raise RuntimeError("Model service not initialized")
            
            models = []
            for model_key, versions in self._model_metadata.items():
                model_symbol, model_type_str = model_key.split("_")
                
                if symbol and model_symbol != symbol:
                    continue
                if model_type and model_type_str != model_type:
                    continue
                
                for version_data in versions:
                    models.append({
                        "symbol": model_symbol,
                        "model_type": model_type_str,
                        "version": version_data["version"],
                        "timestamp": version_data["timestamp"],
                        "metrics": version_data["metrics"],
                        "metadata": version_data["metadata"]
                    })
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def delete_model(
        self,
        symbol: str,
        model_type: str,
        version: str
    ) -> bool:
        """Delete a specific version of a model."""
        try:
            if not self._initialized:
                raise RuntimeError("Model service not initialized")
            
            # Get version directory
            version_dir = self._get_version_dir(symbol, model_type)
            if not version_dir.exists():
                raise FileNotFoundError(f"No model found for {symbol} ({model_type})")
            
            # Delete model files
            model_path = version_dir / f"{symbol}_model.keras"
            metrics_path = version_dir / f"{symbol}_metrics.json"
            metadata_path = version_dir / f"{symbol}_metadata.json"
            
            if model_path.exists():
                model_path.unlink()
            if metrics_path.exists():
                metrics_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update model metadata
            model_key = f"{symbol}_{model_type}"
            if model_key in self._model_metadata:
                self._model_metadata[model_key] = [
                    v for v in self._model_metadata[model_key]
                    if v["version"] != version
                ]
                
                if not self._model_metadata[model_key]:
                    del self._model_metadata[model_key]
                
                await self._save_model_metadata()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model: {str(e)}")
            return False
    
    def _get_version_dir(self, symbol: str, model_type: str) -> Path:
        """Get the directory path for a specific model version."""
        if model_type == "lstm":
            return self.model_dir / symbol
        elif model_type == "prophet":
            return self.prophet_dir / symbol
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_latest_version(self, symbol: str, model_type: str) -> str:
        """Get the latest version of a model."""
        model_key = f"{symbol}_{model_type}"
        if model_key in self._model_metadata and self._model_metadata[model_key]:
            return self._model_metadata[model_key][-1]["version"]
        return self.model_version
    
    async def _load_model_metadata(self) -> None:
        """Load model metadata from disk."""
        metadata_path = self.model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._model_metadata = json.load(f)
    
    async def _save_model_metadata(self) -> None:
        """Save model metadata to disk."""
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._model_metadata, f) 