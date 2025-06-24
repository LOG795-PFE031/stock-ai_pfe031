"""
Service for model storage and versioning.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
from keras import models  # Replace tensorflow import
import joblib

from .base_service import BaseService
from core.logging import logger


class ModelService(BaseService):
    """Service for managing model storage and versioning."""

    def __init__(self):
        super().__init__()
        self.model_dir = Path("models/specific").resolve()
        self.prophet_dir = Path("models/prophet").resolve()
        self._model_metadata = {}
        self.model_version = "1.0.0"
        self.logger = logger["model"]
        self.logger.info(f"Model service initialized with model_dir: {self.model_dir}")
        self.logger.info(
            f"Model service initialized with prophet_dir: {self.prophet_dir}"
        )
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
        metadata: Optional[Dict[str, Any]] = None,
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
            if hasattr(model, "save"):  # Check if it's a Keras model
                models.save_model(model, str(model_path), save_format="keras_v3")
            else:
                model_path = model_dir / f"{symbol}_{model_type}_prophet.joblib"
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

            self._model_metadata[model_key].append(
                {
                    "version": self.model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": metrics,
                    "metadata": metadata or {},
                }
            )

            await self._save_model_metadata()

            return {
                "status": "success",
                "message": f"Model saved successfully for {symbol} ({model_type})",
                "version": self.model_version,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return self.format_error_response(e)

    async def _load_models(self) -> None:
        """Load all models from disk."""
        try:
            loaded_lstm = []
            loaded_prophet = []
            total_processed = 0

            # Get total number of items to process (both directories and files)
            total_items = 0
            if self.model_dir.exists():
                # Count LSTM model directories
                total_items += len(
                    [
                        d
                        for d in self.model_dir.iterdir()
                        if d.is_dir() and not d.name.lower() in ["lstm", "prophet"]
                    ]
                )
            if self.prophet_dir.exists():
                # Count Prophet model files
                total_items += len(list(self.prophet_dir.glob("*_prophet.joblib")))

            # Start loading indicator
            self.logger.info("🤖 Loading models...")

            # Load LSTM models from symbol-specific directories
            if self.model_dir.exists():
                for symbol_dir in self.model_dir.iterdir():
                    if symbol_dir.is_dir() and not symbol_dir.name.lower() in [
                        "lstm",
                        "prophet",
                    ]:
                        symbol = symbol_dir.name
                        try:
                            # Load scaler
                            scaler_path = symbol_dir / f"{symbol}_scaler.gz"
                            model_path = symbol_dir / f"{symbol}_model.keras"

                            if scaler_path.exists() and model_path.exists():
                                self._specific_scalers[symbol] = joblib.load(
                                    scaler_path
                                )
                                self._specific_models[symbol] = models.load_model(
                                    str(model_path), compile=False
                                )
                                loaded_lstm.append(symbol)

                            total_processed += 1
                            progress = (total_processed / total_items) * 100
                            print(
                                f"\r🔄 Loading models... {progress:.1f}% ({total_processed}/{total_items})",
                                end="",
                                flush=True,
                            )

                        except Exception as e:
                            self.logger.error(
                                f"❌ Error loading model for {symbol}: {str(e)}"
                            )

            # Load Prophet models from symbol-specific directories
            prophet_base_dir = self.prophet_dir
            if prophet_base_dir.exists():
                for model_file in prophet_base_dir.glob("*_prophet.joblib"):
                    symbol = model_file.stem.split("_")[0]
                    try:
                        self._specific_models[symbol] = joblib.load(model_file)
                        loaded_prophet.append(symbol)

                        total_processed += 1
                        progress = (total_processed / total_items) * 100
                        print(
                            f"\r🔄 Loading models... {progress:.1f}% ({total_processed}/{total_items})",
                            end="",
                            flush=True,
                        )

                    except Exception as e:
                        self.logger.error(
                            f"❌ Error loading Prophet model for {symbol}: {str(e)}"
                        )

            print()

            self.logger.info(f"✨ Model loading summary:")
            self.logger.info(f"   LSTM models loaded: {len(loaded_lstm)} tickers")
            self.logger.info(f"   Prophet models loaded: {len(loaded_prophet)} tickers")
            if loaded_lstm:
                self.logger.info(f"   LSTM tickers: {', '.join(sorted(loaded_lstm))}")
            if loaded_prophet:
                self.logger.info(
                    f"   Prophet tickers: {', '.join(sorted(loaded_prophet))}"
                )

        except Exception as e:
            self.logger.error(f"❌ Error loading models: {str(e)}")
            raise

    async def load_model(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Load a trained model and its scaler."""
        try:
            self.logger.info(f"Loading {model_type.upper()} model for {symbol}")

            if model_type == "lstm":
                model_dir = self.model_dir / symbol
                model_path = model_dir / f"{symbol}_model.keras"
                scaler_path = model_dir / f"{symbol}_scaler.gz"
                scaler_metadata_path = model_dir / f"{symbol}_scaler_metadata.json"
            elif model_type == "prophet":
                model_dir = self.prophet_dir
                model_path = model_dir / f"{symbol}_prophet.joblib"
                metadata_path = model_dir / f"{symbol}_prophet_metadata.json"
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                return {
                    "status": "error",
                    "error": f"Model file not found for {symbol} ({model_type})",
                    "symbol": symbol,
                    "model_type": model_type,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            try:
                if model_type == "lstm":
                    model = models.load_model(str(model_path), compile=False)
                    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
                    scaler_metadata = {}
                    if scaler_metadata_path.exists():
                        with open(scaler_metadata_path, "r") as f:
                            scaler_metadata = json.load(f)

                    return {
                        "status": "success",
                        "model": model,
                        "scaler": scaler,
                        "scaler_metadata": scaler_metadata,
                        "version": self.model_version,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:  # prophet
                    model = joblib.load(model_path)
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    return {
                        "status": "success",
                        "model": model,
                        "metadata": metadata,
                        "version": self.model_version,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception as e:
                self.logger.error(
                    f"Error loading {model_type} model for {symbol}: {str(e)}"
                )
                return {
                    "status": "error",
                    "error": f"Error loading model: {str(e)}",
                    "symbol": symbol,
                    "model_type": model_type,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error in load_model: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.utcnow().isoformat(),
            }

    # ... (rest of the methods remain unchanged: list_models, delete_model, _get_version_dir, etc.)

    async def list_models(
        self, symbol: Optional[str] = None, model_type: Optional[str] = None
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
                    models.append(
                        {
                            "symbol": model_symbol,
                            "model_type": model_type_str,
                            "version": version_data["version"],
                            "timestamp": version_data["timestamp"],
                            "metrics": version_data["metrics"],
                            "metadata": version_data["metadata"],
                        }
                    )

            return models

        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []

    async def delete_model(self, symbol: str, model_type: str, version: str) -> bool:
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
                    v
                    for v in self._model_metadata[model_key]
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
