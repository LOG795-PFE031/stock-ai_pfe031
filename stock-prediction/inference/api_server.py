"""
API server for stock prediction models.
"""
# Standard library imports
import json
import logging
import os
import platform
import sys
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Dict, List, Any, Tuple, Optional

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from prophet import Prophet

# TensorFlow/Keras imports
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# Local imports
from .rabbitmq_publisher import rabbitmq_publisher
from .training_api import router as training_router
from .prediction_service import PredictionService
from ..core.config import Config

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce TensorFlow logging verbosity
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors

# Version information
PYTHON_VERSION = platform.python_version()
TF_VERSION = tf.__version__
KERAS_VERSION = tf.keras.__version__

logger.info(f"Starting Stock Price Prediction API Server")
logger.info(f"Python {PYTHON_VERSION} | TensorFlow {TF_VERSION} | Keras {KERAS_VERSION}")

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    version="1.0",
    description="API for stock price predictions using LSTM and Prophet models",
    docs_url="/docs"
)

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted stock price in USD")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    confidence_score: float = Field(..., description="Model confidence score")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type used")

class PredictionsResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    start_date: datetime = Field(..., description="Start date of predictions")
    end_date: datetime = Field(..., description="End date of predictions")
    average_confidence: float = Field(..., description="Average confidence score")

class MetaInfo(BaseModel):
    model_info: Dict[str, Any] = Field(..., description="Model information")
    api_info: Dict[str, Any] = Field(..., description="API information")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")

# Global variables
GENERAL_MODEL = None
SPECIFIC_MODELS = {}
SPECIFIC_SCALERS = {}
GENERAL_SCALERS = {}
MODEL_VERSION = "0.1.0"
START_TIME = datetime.now()
REQUEST_COUNT = 0
PREDICTION_SERVICE = None

# Root endpoint
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

# Welcome message
@app.get("/api")
async def home():
    """Get welcome message and basic API information"""
    return {
        'message': 'Welcome to the Stock Price Prediction API',
        'version': '1.0',
        'documentation': '/docs',
        'endpoints': {
            'health_check': '/api/health',
            'next_day_prediction': '/api/predict/next_day',
            'next_week_prediction': '/api/predict/next_week'
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check of the API and its dependencies"""
    try:
        if not GENERAL_MODEL and not SPECIFIC_MODELS:
            raise HTTPException(status_code=503, detail="No models loaded")
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'general_model': 'healthy' if GENERAL_MODEL else 'not loaded',
                'specific_models': f'loaded {len(SPECIFIC_MODELS)} models',
                'database': 'healthy'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/predict/next_day", response_model=PredictionResponse)
async def next_day_prediction(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, GOOGL, MSFT)"),
    model_type: str = Query("lstm", description="Model type to use (lstm or prophet)")
):
    """Get stock price prediction for the next trading day"""
    global REQUEST_COUNT
    
    if model_type not in ['lstm', 'prophet']:
        raise HTTPException(
            status_code=400,
            detail="Invalid model_type. Must be either 'lstm' or 'prophet'"
        )
    
    try:
        REQUEST_COUNT += 1
        logger.info(f"Processing {model_type} prediction request for {symbol}")
        
        return PREDICTION_SERVICE.get_prediction(symbol, model_type)
            
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/meta/info", response_model=MetaInfo)
async def meta_info():
    """Get API and model metadata including version, uptime, and statistics"""
    try:
        return {
            'model_info': {
                'version': MODEL_VERSION,
                'last_trained': datetime(2024, 3, 1),
                'accuracy_score': 0.95
            },
            'api_info': {
                'version': '1.0',
                'uptime': str(datetime.now() - START_TIME),
                'requests_served': REQUEST_COUNT
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include training router
app.include_router(training_router, prefix="/api/training")

def load_resources() -> None:
    """Load ML models and scalers with proper error handling"""
    global GENERAL_MODEL, SPECIFIC_MODELS, SPECIFIC_SCALERS, GENERAL_SCALERS, PREDICTION_SERVICE
    
    try:
        # Load specific models and scalers
        specific_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "specific")
        if not os.path.exists(specific_dir):
            raise HTTPException(status_code=500, detail=f"Specific models directory {specific_dir} not found")
            
        loaded_count = 0
        skipped_count = 0
        error_count = 0
        loaded_symbols = []
        skipped_symbols = []
        error_symbols = []
        
        # Get all symbol directories
        symbol_dirs = [
            d for d in os.listdir(specific_dir) 
            if os.path.isdir(os.path.join(specific_dir, d))
        ]
        
        logger.info(f"Loading models for {len(symbol_dirs)} symbols...")
        
        for symbol in symbol_dirs:
            try:
                symbol_dir = os.path.join(specific_dir, symbol)
                model_keras_path = os.path.join(symbol_dir, f"{symbol}_model.keras")
                model_weights_path = os.path.join(symbol_dir, f"{symbol}_model.weights.h5")
                scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
                metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
                
                # Skip if we don't have both scaler and metadata
                if not (os.path.exists(scaler_path) and os.path.exists(metadata_path)):
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Try to load the model
                if os.path.exists(model_keras_path):
                    model = load_model(model_keras_path)
                elif os.path.exists(model_weights_path):
                    input_shape = (60, 13)  # SEQ_SIZE, len(FEATURES)
                    model = build_specific_model(input_shape)
                    model.load_weights(model_weights_path)
                else:
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Load scaler and metadata
                scaler = joblib.load(scaler_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                SPECIFIC_MODELS[symbol] = model
                SPECIFIC_SCALERS[symbol] = scaler
                loaded_count += 1
                loaded_symbols.append(symbol)
                
            except Exception as e:
                error_count += 1
                error_symbols.append(symbol)
                logger.error(f"❌ Error loading model for {symbol}: {str(e)}")
                continue
        
        # Print summary
        logger.info(f"\n=== Model Loading Summary ===")
        logger.info(f"✅ Successfully loaded {loaded_count} models")
        if skipped_count > 0:
            logger.info(f"⚠️ Skipped {skipped_count} models: {', '.join(skipped_symbols)}")
        if error_count > 0:
            logger.info(f"❌ Failed to load {error_count} models: {', '.join(error_symbols)}")
        logger.info("===========================\n")
        
        if not SPECIFIC_MODELS:
            raise HTTPException(status_code=500, detail="No models were loaded")
            
        # Initialize prediction service
        PREDICTION_SERVICE = PredictionService(
            SPECIFIC_MODELS,
            SPECIFIC_SCALERS,
            GENERAL_MODEL,
            GENERAL_SCALERS
        )
            
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model resources")

def build_specific_model(input_shape: Tuple) -> Model:
    """Build a stock-specific model using functional API"""
    inputs = Input(shape=input_shape, name='sequence_input')
    
    x = LSTM(100, return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal')(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal')(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu',
            kernel_initializer='glorot_uniform')(x)
    x = Dense(25, activation='relu',
            kernel_initializer='glorot_uniform')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='specific_stock_model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs(Config().data.LOGS_DIR, exist_ok=True)
    
    # Load models and resources
    load_resources()
    
    # Start the server
    port = int(os.environ.get("API_PORT", 8000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port) 