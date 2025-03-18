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
from typing import Dict, List, Any, Tuple

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request
from flask_restx import Api, Resource, fields
from prophet import Prophet

# TensorFlow/Keras imports
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# Local imports
from .rabbitmq_publisher import rabbitmq_publisher
from .training_api import ns as training_ns
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

app = Flask(__name__)
api = Api(
    app, 
    title='Stock Price Prediction API',
    version='1.0',
    description='A production-ready API for predicting stock prices using LSTM and Prophet',
    doc='/docs',
    default='Stock Prediction API',
    default_label='ML-powered stock price prediction endpoints'
)

# Create namespaces for better organization
ns_home = api.namespace('', description='Home page')
ns_predict = api.namespace('predict', description='Stock price prediction operations')
ns_health = api.namespace('health', description='Health and monitoring operations')
ns_meta = api.namespace('meta', description='API metadata and information')

# Add training namespace
api.add_namespace(training_ns)

# Response Models
error_model = api.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'status_code': fields.Integer(required=True, description='HTTP status code'),
    'timestamp': fields.DateTime(required=True, description='Error timestamp')
})

prediction_model = api.model('Prediction', {
    'prediction': fields.Float(required=True, description='Predicted stock price in USD'),
    'timestamp': fields.DateTime(required=True, description='Prediction timestamp'),
    'confidence_score': fields.Float(required=True, description='Model confidence score'),
    'model_version': fields.String(required=True, description='Model version'),
    'model_type': fields.String(required=True, description='Model type used')
})

predictions_model = api.model('Predictions', {
    'predictions': fields.List(
        fields.Nested(prediction_model), 
        description='List of predictions'
    ),
    'start_date': fields.DateTime(description='Start date of predictions'),
    'end_date': fields.DateTime(description='End date of predictions'),
    'average_confidence': fields.Float(description='Average confidence score')
})

meta_model = api.model('MetaInfo', {
    'model_info': fields.Nested(api.model('ModelInfo', {
        'version': fields.String(description='Model version'),
        'last_trained': fields.DateTime(description='Last training timestamp'),
        'accuracy_score': fields.Float(description='Model accuracy score')
    })),
    'api_info': fields.Nested(api.model('ApiInfo', {
        'version': fields.String(description='API version'),
        'uptime': fields.String(description='API uptime'),
        'requests_served': fields.Integer(description='Total requests served')
    }))
})

# Global variables
GENERAL_MODEL = None
SPECIFIC_MODELS = {}
SPECIFIC_SCALERS = {}
GENERAL_SCALERS = {}
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()
REQUEST_COUNT = 0
PREDICTION_SERVICE = None

class ModelNotLoadedError(Exception):
    """Raised when the model is not properly loaded"""
    pass

def load_resources() -> None:
    """Load ML models and scalers with proper error handling"""
    global GENERAL_MODEL, SPECIFIC_MODELS, SPECIFIC_SCALERS, GENERAL_SCALERS, PREDICTION_SERVICE
    
    try:
        # Load specific models and scalers
        specific_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "specific")
        if not os.path.exists(specific_dir):
            raise ModelNotLoadedError(f"Specific models directory {specific_dir} not found")
            
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
            raise ModelNotLoadedError("No models were loaded")
            
        # Initialize prediction service
        PREDICTION_SERVICE = PredictionService(
            SPECIFIC_MODELS,
            SPECIFIC_SCALERS,
            GENERAL_MODEL,
            GENERAL_SCALERS
        )
            
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        raise ModelNotLoadedError("Failed to load model resources") from e

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

@ns_home.route('/')
class Home(Resource):
    @api.doc(
        responses={
            200: 'Welcome message',
        }
    )
    def get(self):
        """Get welcome message and basic API information"""
        return {
            'message': 'Welcome to the Stock Price Prediction API',
            'version': '1.0',
            'documentation': '/docs',
            'endpoints': {
                'health_check': '/health',
                'next_day_prediction': '/predict/next_day',
                'next_week_prediction': '/predict/next_week'
            }
        }

@ns_meta.route('/info')
class MetaInfo(Resource):
    @api.doc(
        description='Get API and model metadata',
        responses={
            HTTPStatus.OK: ('Success', meta_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Server Error', error_model)
        }
    )
    @api.marshal_with(meta_model)
    def get(self) -> Dict[str, Any]:
        """Get API and model metadata including version, uptime, and statistics"""
        global REQUEST_COUNT
        try:
            return {
                'model_info': {
                    'version': MODEL_VERSION,
                    'last_trained': datetime(2024, 3, 1),  # Example date
                    'accuracy_score': 0.95  # Example score
                },
                'api_info': {
                    'version': '1.0',
                    'uptime': str(datetime.now() - START_TIME),
                    'requests_served': REQUEST_COUNT
                }
            }
        except Exception as e:
            api.abort(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

@ns_health.route('/')
class HealthCheck(Resource):
    @api.doc(
        description='Check API health status',
        responses={
            HTTPStatus.OK: 'API is healthy',
            HTTPStatus.SERVICE_UNAVAILABLE: 'API is unhealthy'
        }
    )
    def get(self) -> Dict[str, Any]:
        """Comprehensive health check of the API and its dependencies"""
        try:
            # Verify models are loaded
            if not GENERAL_MODEL and not SPECIFIC_MODELS:
                raise ModelNotLoadedError("No models loaded")
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'general_model': 'healthy' if GENERAL_MODEL else 'not loaded',
                    'specific_models': f'loaded {len(SPECIFIC_MODELS)} models',
                    'database': 'healthy'
                }
            }, HTTPStatus.OK
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, HTTPStatus.SERVICE_UNAVAILABLE

@ns_predict.route('/next_day')
class NextDayPrediction(Resource):
    @api.doc(
        description='Get stock price prediction for the next trading day',
        params={
            'symbol': {
                'in': 'query',
                'description': 'Stock symbol (e.g., AAPL, GOOGL, MSFT)',
                'type': 'string',
                'required': True
            },
            'model_type': {
                'in': 'query',
                'description': 'Model type to use for prediction (lstm or prophet)',
                'type': 'string',
                'enum': ['lstm', 'prophet'],
                'default': 'lstm'
            }
        },
        responses={
            HTTPStatus.OK: ('Successful prediction', prediction_model),
            HTTPStatus.BAD_REQUEST: ('Invalid request', error_model),
            HTTPStatus.NOT_FOUND: ('Model not found', error_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Prediction error', error_model)
        }
    )
    @api.marshal_with(prediction_model)
    def get(self) -> Dict[str, Any]:
        """Get detailed stock price prediction for the next trading day"""
        global REQUEST_COUNT
        
        # Get stock symbol and model type from query parameters
        stock_symbol = request.args.get('symbol') or request.headers.get('X-Fields')
        model_type = request.args.get('model_type', 'lstm').lower()
        
        if not stock_symbol:
            api.abort(
                HTTPStatus.BAD_REQUEST,
                "Stock symbol is required. Provide it as a query parameter 'symbol' or header 'X-Fields'"
            )
            
        if model_type not in ['lstm', 'prophet']:
            api.abort(
                HTTPStatus.BAD_REQUEST,
                "Invalid model_type. Must be either 'lstm' or 'prophet'"
            )
        
        try:
            REQUEST_COUNT += 1
            logger.info(f"Processing {model_type} prediction request for {stock_symbol}")
            
            return PREDICTION_SERVICE.get_prediction(stock_symbol, model_type)
                
        except Exception as e:
            logger.error(f"Prediction error for {stock_symbol}: {str(e)}")
            api.abort(
                HTTPStatus.INTERNAL_SERVER_ERROR, 
                f"Prediction error: {str(e)}"
            )

@ns_predict.route('/debug_scaling/<string:symbol>')
class ScalingDebug(Resource):
    def get(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about scaling operations for debugging"""
        try:
            # Get the latest sequence
            sequence = PREDICTION_SERVICE._get_latest_sequence(symbol)
            
            # Try to load scaling metadata
            metadata_path = os.path.join("models", "specific", symbol, f"{symbol}_scaler_metadata.json")
            scaling_info = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    scaling_metadata = json.load(f)
                scaling_info['metadata'] = scaling_metadata
            
            # Get scaler
            if symbol in SPECIFIC_MODELS:
                scaler = SPECIFIC_SCALERS[symbol]
                model_type = "specific"
            elif GENERAL_MODEL:
                scaler = GENERAL_SCALERS['symbol']
                model_type = "general"
            else:
                return {'error': 'No model available'}, HTTPStatus.NOT_FOUND
            
            # Get last known values
            last_sequence = sequence[0, -1, :]
            close_index = PREDICTION_SERVICE.features.index("Close")
            last_close = last_sequence[close_index]
            
            # Make a test prediction
            model = SPECIFIC_MODELS[symbol] if symbol in SPECIFIC_MODELS else GENERAL_MODEL
            prediction = model.predict(sequence)
            
            # Get original price
            original_price = PREDICTION_SERVICE._get_original_price(symbol)
            
            # Try both scaling methods
            scaling_info['debug'] = {
                'model_type': model_type,
                'sequence_shape': sequence.shape,
                'last_known_normalized_close': float(last_close),
                'original_price': float(original_price) if original_price else None,
                'raw_prediction': float(prediction[0, 0]),
                'features_available': PREDICTION_SERVICE.features,
                'scaler_feature_names': scaling_metadata['feature_names'] if 'metadata' in scaling_info else None
            }
            
            return scaling_info
            
        except Exception as e:
            return {'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config().data.LOGS_DIR, exist_ok=True)
    
    # Load models and resources
    load_resources()
    
    # Start the server
    app.run(host='0.0.0.0', port=8000) 