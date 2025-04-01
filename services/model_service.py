"""
Service for model storage and versioning.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import shutil
from pathlib import Path
import tensorflow as tf
import joblib
from packaging import version

from services.base_service import BaseService
from core.config import config
from core.logging import logger

class ModelService(BaseService):
    """Service for managing model storage and versioning."""
    
    def __init__(self):
        super().__init__()
        self.model_dir = config.model.PREDICTION_MODELS_DIR
        self._model_metadata = {}
        self.model_version = "1.0.0"
        self.logger = logger['model']
    
    async def initialize(self) -> None:
        """Initialize the model service."""
        try:
            await self._load_model_metadata()
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
                model.save(model_path)
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
    
    async def load_model(
        self,
        symbol: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a trained model with its metrics and metadata."""
        try:
            if not self._initialized:
                raise RuntimeError("Model service not initialized")
            
            # Get version to load
            if not version:
                version = self._get_latest_version(symbol, model_type)
            
            # Get version directory
            version_dir = self._get_version_dir(symbol, model_type)
            if not version_dir.exists():
                raise FileNotFoundError(f"No model found for {symbol} ({model_type})")
            
            # Load model
            model_path = version_dir / f"{symbol}_model.keras"
            if not model_path.exists():
                # Try alternative model paths
                model_paths = [
                    version_dir / f"{symbol}_model.keras",
                    version_dir / f"{symbol}_{model_type}_model.keras",
                    version_dir / f"{symbol}_model.weights.h5"
                ]
                model_path = next((p for p in model_paths if p.exists()), None)
                if not model_path:
                    raise FileNotFoundError(f"Model file not found for {symbol} ({model_type})")
            
            # Load metrics
            metrics_path = version_dir / f"{symbol}_metrics.json"
            if not metrics_path.exists():
                # Try alternative metrics paths
                metrics_paths = [
                    version_dir / f"{symbol}_metrics.json",
                    version_dir / f"{symbol}_{model_type}_metrics.json"
                ]
                metrics_path = next((p for p in metrics_paths if p.exists()), None)
                if not metrics_path:
                    raise FileNotFoundError(f"Metrics file not found for {symbol} ({model_type})")
            
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            # Load metadata if exists
            metadata = {}
            metadata_path = version_dir / f"{symbol}_metadata.json"
            if not metadata_path.exists():
                # Try alternative metadata paths
                metadata_paths = [
                    version_dir / f"{symbol}_metadata.json",
                    version_dir / f"{symbol}_{model_type}_metadata.json",
                    version_dir / f"{symbol}_scaler_metadata.json"
                ]
                metadata_path = next((p for p in metadata_paths if p.exists()), None)
                if metadata_path:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
            
            # Load model
            try:
                if model_path.suffix == '.keras':
                    model = tf.keras.models.load_model(model_path)
                elif model_path.suffix == '.h5':
                    # Load weights file
                    model = tf.keras.models.load_model(version_dir / f"{symbol}_model.keras")
                    model.load_weights(model_path)
                else:
                    model = joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise
            
            # Load scaler
            scaler_path = version_dir / f"{symbol}_scaler.gz"
            if not scaler_path.exists():
                # Try alternative scaler paths
                scaler_paths = [
                    version_dir / f"{symbol}_scaler.gz",
                    version_dir / f"{symbol}_{model_type}_scaler.gz"
                ]
                scaler_path = next((p for p in scaler_paths if p.exists()), None)
                if not scaler_path:
                    raise FileNotFoundError(f"Scaler file not found for {symbol} ({model_type})")
            
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                self.logger.error(f"Error loading scaler: {str(e)}")
                raise
            
            return {
                "status": "success",
                "model": model,
                "scaler": scaler,
                "metrics": metrics,
                "metadata": metadata,
                "version": version,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return self.format_error_response(e)
    
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
        """Get the directory for a specific model version."""
        # Convert symbol to uppercase for consistency
        symbol = symbol.upper()
        # Models are stored directly in symbol-specific directories
        symbol_dir = self.model_dir / symbol
        return symbol_dir
    
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