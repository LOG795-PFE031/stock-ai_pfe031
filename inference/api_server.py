from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model, Model
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Dict, List, Any, Tuple
import logging
import os
import tensorflow as tf
import sys
import platform
from prophet import Prophet
from .rabbitmq_publisher import rabbitmq_publisher
import json
from keras.layers import LSTM, Dropout, Dense, Input
from functools import wraps

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
    description='A production-ready API for predicting stock prices using LSTM',
    doc='/docs',
    default='Stock Prediction API',
    default_label='ML-powered stock price prediction endpoints'
)

# Create namespaces for better organization
ns_home = api.namespace('', description='Home page')
ns_predict = api.namespace('predict', description='Stock price prediction operations')
ns_health = api.namespace('health', description='Health and monitoring operations')
ns_meta = api.namespace('meta', description='API metadata and information')

# Response Models
error_model = api.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'status_code': fields.Integer(required=True, description='HTTP status code'),
    'timestamp': fields.DateTime(required=True, description='Error timestamp')
})

prediction_model = api.model('Prediction', {
    'prediction': fields.Float(required=True, description='Predicted stock price in USD'),
    'timestamp': fields.DateTime(required=True, description='Prediction timestamp'),
    'confidence_score': fields.Float(description='Model confidence score (0-1)', min=0, max=1),
    'model_version': fields.String(description='Version of the ML model used'),
    'model_type': fields.String(description='Type of model used (general or specific)')
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
PROPHET_MODELS = {}  # Store Prophet models for each symbol
SEQ_SIZE = 60
FEATURES = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Returns", "MA_5", "MA_20", "Volatility", "RSI", "MACD", "MACD_Signal"
]
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()
REQUEST_COUNT = 0

class ModelNotLoadedError(Exception):
    """Raised when the model is not properly loaded"""
    pass

def load_prophet_model(symbol: str) -> Prophet:
    """Load or create a Prophet model for a given symbol"""
    try:
        # Check if model already exists in memory
        if symbol in PROPHET_MODELS:
            return PROPHET_MODELS[symbol]
            
        # Try to load from disk
        prophet_model_path = os.path.join("models", "prophet", f"{symbol}_prophet.json")
        if os.path.exists(prophet_model_path):
            with open(prophet_model_path, 'r') as fin:
                model_data = json.load(fin)
            
            # Create model with saved parameters
            model = Prophet(**model_data['params'])
            
            # Add regressors
            for regressor in model_data['regressors']:
                model.add_regressor(regressor)
            
            # Convert last data back to DataFrame
            df = pd.DataFrame(model_data['last_data'])
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Fit the model on the last data
            model.fit(df)
            
            PROPHET_MODELS[symbol] = model
            logger.info(f"✅ Loaded Prophet model for {symbol}")
            return model
            
        # Create new model if not found
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        PROPHET_MODELS[symbol] = model
        logger.info(f"✅ Created new Prophet model for {symbol}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading Prophet model for {symbol}: {str(e)}")
        raise ModelNotLoadedError(f"Failed to load Prophet model for {symbol}") from e

def load_model_safely(model_path: str, symbol: str) -> Model:
    """Load a Keras model with additional safety measures for compatibility"""
    try:
        # First try normal loading
        logger.info(f"Attempting to load model for {symbol} with standard method")
        return load_model(model_path)
    except Exception as e:
        logger.warning(f"Standard loading failed for {symbol}: {str(e)}")
        
        # Try with unsafe deserialization explicitly enabled
        try:
            logger.info(f"Attempting to load model for {symbol} with unsafe deserialization")
            # Ensure unsafe deserialization is enabled
            tf.keras.config.enable_unsafe_deserialization()
            return load_model(model_path)
        except Exception as unsafe_error:
            logger.warning(f"Unsafe loading failed for {symbol}: {str(unsafe_error)}")
            
            # Try with custom objects
            try:
                logger.info(f"Attempting to load model for {symbol} with custom objects")
                
                # Define common custom objects that might be needed
                custom_objects = {
                    'LSTM': tf.keras.layers.LSTM,
                    'Dropout': tf.keras.layers.Dropout,
                    'Dense': tf.keras.layers.Dense,
                    'Input': tf.keras.layers.Input
                }
                
                return load_model(model_path, custom_objects=custom_objects)
            except Exception as custom_error:
                logger.error(f"All loading methods failed for {symbol}")
                raise Exception(f"Could not load model: {str(custom_error)}") from custom_error

def load_resources() -> None:
    """Load ML models and scalers with proper error handling"""
    global GENERAL_MODEL, SPECIFIC_MODELS, SPECIFIC_SCALERS, GENERAL_SCALERS
    
    try:
        # Load specific models and scalers
        specific_dir = "models/specific"
        if not os.path.exists(specific_dir):
            raise ModelNotLoadedError(f"Specific models directory {specific_dir} not found")
            
        loaded_count = 0
        skipped_count = 0
        error_count = 0
        loaded_symbols = []
        skipped_symbols = []
        error_symbols = []
        missing_metadata_count = 0
        missing_scaler_count = 0
        tf_compatibility_errors = 0
        
        # Get all symbol directories
        symbol_dirs = [
            d for d in os.listdir(specific_dir) 
            if os.path.isdir(os.path.join(specific_dir, d))
        ]
        
        logger.info(f"Loading models for {len(symbol_dirs)} symbols...")
        logger.info(f"TensorFlow version: {tf.__version__}, Keras version: {tf.keras.__version__}")
        
        for symbol in symbol_dirs:
            try:
                symbol_dir = os.path.join(specific_dir, symbol)
                model_keras_path = os.path.join(symbol_dir, f"{symbol}_model.keras")
                model_weights_path = os.path.join(symbol_dir, f"{symbol}_model.weights.h5")
                scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
                metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
                
                # Add detailed logging
                file_status = {
                    "model_keras": os.path.exists(model_keras_path),
                    "model_weights": os.path.exists(model_weights_path),
                    "scaler": os.path.exists(scaler_path),
                    "metadata": os.path.exists(metadata_path)
                }
                
                logger.info(f"Files for {symbol}: {file_status}")
                
                # Skip if we don't have both scaler and metadata
                if not (os.path.exists(scaler_path) and os.path.exists(metadata_path)):
                    # Track what's missing
                    if not os.path.exists(scaler_path):
                        missing_scaler_count += 1
                        skipped_count += 1
                        skipped_symbols.append(symbol)
                        continue
                    elif not os.path.exists(metadata_path):
                        missing_metadata_count += 1
                        # Try to create metadata on the fly if scaler exists
                        logger.info(f"Attempting to create missing metadata for {symbol}")
                        if create_metadata_for_symbol(symbol):
                            logger.info(f"Successfully created metadata for {symbol}")
                            # Now the metadata exists, so we can proceed
                        else:
                            skipped_count += 1
                            skipped_symbols.append(symbol)
                            continue
                
                # Try to load the model
                if os.path.exists(model_keras_path):
                    try:
                        logger.info(f"Loading .keras model for {symbol} from {model_keras_path}")
                        # Check if model was saved with a different TF/Keras version
                        try:
                            with open(model_keras_path, 'rb') as f:
                                # Just try to read first few bytes to see if file is accessible
                                f.read(10)
                            logger.info(f"Model file for {symbol} is readable")
                        except Exception as file_error:
                            logger.error(f"Cannot read model file for {symbol}: {str(file_error)}")
                        
                        model = load_model_safely(model_keras_path, symbol)
                    except Exception as e:
                        logger.error(f"❌ Error loading model for {symbol}: {str(e)}")
                        tf_compatibility_errors += 1
                        error_count += 1
                        error_symbols.append(symbol)
                        continue
                elif os.path.exists(model_weights_path):
                    logger.info(f"Loading model weights for {symbol} from {model_weights_path}")
                    try:
                        input_shape = (SEQ_SIZE, len(FEATURES))
                        model = build_specific_model(input_shape)
                        model.load_weights(model_weights_path)
                    except Exception as e:
                        logger.error(f"❌ Error loading model weights for {symbol}: {str(e)}")
                        tf_compatibility_errors += 1
                        error_count += 1
                        error_symbols.append(symbol)
                        continue
                else:
                    logger.info(f"No model file found for {symbol}, skipping")
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Load scaler and metadata
                try:
                    scaler = joblib.load(scaler_path)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        logger.info(f"Successfully loaded metadata for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Error loading scaler or metadata for {symbol}: {str(e)}")
                    error_count += 1
                    error_symbols.append(symbol)
                    continue
                
                SPECIFIC_MODELS[symbol] = model
                SPECIFIC_SCALERS[symbol] = scaler
                loaded_count += 1
                loaded_symbols.append(symbol)
                logger.info(f"✅ Successfully loaded model, scaler and metadata for {symbol}")
                
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
        
        # Print detailed statistics
        logger.info(f"Missing metadata files: {missing_metadata_count}")
        logger.info(f"Missing scaler files: {missing_scaler_count}")
        logger.info(f"TensorFlow/Keras compatibility errors: {tf_compatibility_errors}")
        logger.info("===========================\n")
        
        if not SPECIFIC_MODELS:
            raise ModelNotLoadedError("No models were loaded")
            
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        raise ModelNotLoadedError("Failed to load model resources") from e

def create_metadata_for_symbol(symbol: str) -> bool:
    """Create metadata file for a specific symbol if scaler exists but metadata doesn't"""
    specific_dir = "models/specific"
    symbol_dir = os.path.join(specific_dir, symbol)
    
    if not os.path.isdir(symbol_dir):
        logger.warning(f"No directory found for symbol {symbol}")
        return False
        
    scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
    metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
    
    # Skip if metadata exists or scaler doesn't exist
    if os.path.exists(metadata_path):
        logger.info(f"Metadata already exists for {symbol}")
        return False
        
    if not os.path.exists(scaler_path):
        logger.warning(f"No scaler found for {symbol}")
        return False
        
    try:
        # Load the scaler
        scaler = joblib.load(scaler_path)
        
        # Create basic metadata with default values if actual values can't be extracted
        try:
            min_values = scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else [0.0] * len(FEATURES)
            max_values = scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else [1.0] * len(FEATURES)
        except Exception as e:
            logger.warning(f"Could not extract min/max values from scaler for {symbol}: {str(e)}")
            min_values = [0.0] * len(FEATURES)
            max_values = [1.0] * len(FEATURES)
        
        scaling_metadata = {
            'feature_order': FEATURES,
            'min_values': min_values,
            'max_values': max_values,
            'feature_names': FEATURES
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(scaling_metadata, f, indent=4)
            
        logger.info(f"Created metadata file for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Failed to create metadata for {symbol}: {str(e)}")
        return False

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

def get_latest_sequence(symbol: str) -> np.ndarray:
    """Retrieves and prepares the most recent sequence for a specific stock"""
    try:
        # Try to find the stock in the specific directory structure
        found_file = False
        try:
            for sector in os.listdir(os.path.join("data", "processed", "specific")):
                sector_path = os.path.join("data", "processed", "specific", sector)
                if os.path.isdir(sector_path):
                    stock_file = os.path.join(sector_path, f"{symbol}_processed.csv")
                    if os.path.exists(stock_file):
                        df = pd.read_csv(stock_file)
                        found_file = True
                        logger.info(f"Found data file for {symbol} in {sector_path}")
                        break
        except Exception as e:
            logger.error(f"Error searching for {symbol} in specific sectors: {str(e)}")
                    
        if not found_file:
            # If not found in specific directories, try the general processed directory
            stock_file = os.path.join("data", "processed", f"{symbol}_processed.csv")
            if not os.path.exists(stock_file):
                raise FileNotFoundError(f"No data found for symbol {symbol}")
            df = pd.read_csv(stock_file)
            logger.info(f"Found data file for {symbol} in general processed directory")
        
        # Get last SEQ_SIZE samples
        if len(df) < SEQ_SIZE:
            logger.warning(f"Not enough data for {symbol}. Need {SEQ_SIZE} samples, but only have {len(df)}.")
            # Pad the data by repeating the first row
            padding = pd.concat([df.iloc[[0]]] * (SEQ_SIZE - len(df)), ignore_index=True)
            df = pd.concat([padding, df], ignore_index=True)
            
        df = df.tail(SEQ_SIZE)
        
        # Check if all required features are present
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features for {symbol}: {missing_features}")
            logger.error(f"Available features: {list(df.columns)}")
            raise ValueError(f"Missing required features for {symbol}: {missing_features}")
        
        # Prepare sequence with all required features
        sequence = df[FEATURES].values.reshape(1, SEQ_SIZE, len(FEATURES))
        logger.info(f"Successfully prepared sequence for {symbol} with shape {sequence.shape}")
        return sequence
        
    except FileNotFoundError as e:
        logger.error(f"File not found error for {symbol}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error getting latest sequence for {symbol}: {str(e)}")
        logger.exception("Full traceback:")
        raise

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Global error handler caught: {str(e)}")
    logger.exception("Unhandled error:")
    return {
        'error': str(e),
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }, HTTPStatus.INTERNAL_SERVER_ERROR

# Route protection wrapper
def handle_route_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Route error: {str(e)}")
            logger.exception("Full traceback:")
            return {
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    return decorated

@ns_home.route('/')
class Home(Resource):
    @api.doc(
        responses={
            200: 'Welcome message',
        }
    )
    @handle_route_errors
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
    @handle_route_errors
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
    @handle_route_errors
    def get(self) -> Dict[str, Any]:
        """Comprehensive health check of the API and its dependencies"""
        try:
            # Verify models are loaded
            if not GENERAL_MODEL and not SPECIFIC_MODELS:
                raise ModelNotLoadedError("No models loaded")
            
            # Check data directories
            data_dir_status = {}
            
            try:
                if os.path.exists("data"):
                    data_dir_status["data_root"] = "exists"
                    
                    if os.path.exists("data/processed"):
                        data_dir_status["processed"] = "exists"
                        
                        if os.path.exists("data/processed/specific"):
                            data_dir_status["processed/specific"] = "exists"
                            specific_dirs = os.listdir("data/processed/specific")
                            data_dir_status["sectors"] = specific_dirs
            except Exception as e:
                logger.error(f"Error checking data directories: {str(e)}")
                data_dir_status["error"] = str(e)
            
            # Check specific models availability
            model_status = {}
            try:
                model_status["total_models"] = len(SPECIFIC_MODELS)
                model_status["model_keys"] = list(SPECIFIC_MODELS.keys())[:5]  # First 5 models
                model_status["total_scalers"] = len(SPECIFIC_SCALERS)
                model_status["scaler_keys"] = list(SPECIFIC_SCALERS.keys())[:5]  # First 5 scalers
            except Exception as e:
                logger.error(f"Error checking model status: {str(e)}")
                model_status["error"] = str(e)
            
            # Create the health status response
            result = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'general_model': 'healthy' if GENERAL_MODEL else 'not loaded',
                    'specific_models': model_status,
                    'data_directories': data_dir_status,
                    'system_info': {
                        'python_version': PYTHON_VERSION,
                        'tensorflow_version': TF_VERSION,
                        'keras_version': KERAS_VERSION
                    }
                }
            }
            logger.info("Health check successful")
            return result, HTTPStatus.OK
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.exception("Full error traceback:")
            error_result = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'python_version': PYTHON_VERSION,
                    'tensorflow_version': TF_VERSION,
                    'keras_version': KERAS_VERSION
                }
            }
            return error_result, HTTPStatus.SERVICE_UNAVAILABLE

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
    @handle_route_errors
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
            
            if model_type == 'prophet':
                return self._get_prophet_prediction(stock_symbol)
            else:
                return self._get_lstm_prediction(stock_symbol)
                
        except Exception as e:
            logger.error(f"Prediction error for {stock_symbol}: {str(e)}")
            api.abort(
                HTTPStatus.INTERNAL_SERVER_ERROR, 
                f"Prediction error: {str(e)}"
            )
            
    def _get_prophet_prediction(self, stock_symbol: str) -> Dict[str, Any]:
        """Get prediction using Prophet model"""
        try:
            # Load or create Prophet model
            model = load_prophet_model(stock_symbol)
            
            # Get historical data for the stock
            stock_file = None
            
            # First check in Technology sector
            tech_file = os.path.join("data", "raw", "Technology", f"{stock_symbol}_stock_price.csv")
            if os.path.exists(tech_file):
                stock_file = tech_file
            
            # Then check in processed/specific directories
            if stock_file is None:
                for sector in os.listdir(os.path.join("data", "processed", "specific")):
                    sector_path = os.path.join("data", "processed", "specific", sector)
                    if os.path.isdir(sector_path):
                        temp_file = os.path.join(sector_path, f"{stock_symbol}_processed.csv")
                        if os.path.exists(temp_file):
                            stock_file = temp_file
                            break
            
            # Finally check in raw data root
            if stock_file is None:
                raw_file = os.path.join("data", "raw", f"{stock_symbol}_stock_price.csv")
                if os.path.exists(raw_file):
                    stock_file = raw_file
            
            if stock_file is None:
                raise FileNotFoundError(f"No data found for symbol {stock_symbol}")
            
            logger.info(f"Using data file: {stock_file}")
            
            # Load and prepare data
            df = pd.read_csv(stock_file)
            df['ds'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
            df['y'] = df['Close']
            
            # Prepare regressors
            regressors = []
            if 'Volume' in df.columns:
                df['volume'] = df['Volume'].fillna(0)
                regressors.append('volume')
                
            if 'RSI' in df.columns:
                df['rsi'] = df['RSI'].fillna(df['RSI'].mean())
                regressors.append('rsi')
            
            # Make prediction
            future = model.make_future_dataframe(periods=1)
            
            # Add regressors to future dataframe
            if 'volume' in regressors:
                # Use the last known volume for prediction
                future['volume'] = df['volume'].iloc[-1]
                
            if 'rsi' in regressors:
                # Use the last known RSI for prediction
                future['rsi'] = df['rsi'].iloc[-1]
            
            forecast = model.predict(future)
            
            # Get the prediction for tomorrow
            prediction = forecast.iloc[-1]['yhat']
            
            # Calculate confidence score based on uncertainty intervals
            lower = forecast.iloc[-1]['yhat_lower']
            upper = forecast.iloc[-1]['yhat_upper']
            interval_width = upper - lower
            last_price = df['Close'].iloc[-1]
            confidence_score = max(0, min(1, 1 - (interval_width / (4 * last_price))))
            
            result = {
                'prediction': float(prediction),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': MODEL_VERSION,
                'model_type': 'prophet'
            }
            
            # Publish to RabbitMQ
            try:
                publish_success = rabbitmq_publisher.publish_stock_quote(stock_symbol, result)
                if publish_success:
                    logger.info(f"✅ Successfully published Prophet prediction for {stock_symbol} to RabbitMQ")
                    result['rabbitmq_status'] = 'delivered'
                else:
                    logger.warning(f"⚠️ Failed to confirm RabbitMQ delivery for {stock_symbol}")
                    result['rabbitmq_status'] = 'unconfirmed'
            except Exception as e:
                logger.error(f"❌ Failed to publish to RabbitMQ: {str(e)}")
                result['rabbitmq_status'] = 'failed'
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet prediction error for {stock_symbol}: {str(e)}")
            raise
            
    def _get_lstm_prediction(self, stock_symbol: str) -> Dict[str, Any]:
        """Get prediction using LSTM model"""
        try:
            # Get the latest sequence
            sequence = get_latest_sequence(stock_symbol)
            
            # Try to use specific model first
            if stock_symbol in SPECIFIC_MODELS:
                model = SPECIFIC_MODELS[stock_symbol]
                scaler = SPECIFIC_SCALERS[stock_symbol]
                model_type = "specific"
                
                # Load scaling metadata if available
                metadata_path = os.path.join("models", "specific", stock_symbol, f"{stock_symbol}_scaler_metadata.json")
                scaling_metadata = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        scaling_metadata = json.load(f)
                logger.info(f"Using specific model for {stock_symbol}")
            # Fall back to general model if available
            elif GENERAL_MODEL:
                model = GENERAL_MODEL
                scaler = GENERAL_SCALERS['symbol']
                model_type = "general"
                logger.info(f"Using general model for {stock_symbol}")
            else:
                api.abort(
                    HTTPStatus.NOT_FOUND,
                    f"No model available for {stock_symbol}"
                )
            
            # Make prediction
            prediction = model.predict(sequence)
            logger.info(f"Raw prediction shape: {prediction.shape}, values: {prediction}")
            
            # Get the last known values
            last_sequence = sequence[0, -1, :]  # Get the last row of the sequence
            
            # Get the index of the "Close" feature
            close_index = FEATURES.index("Close")
            
            # Get the last known close price (still normalized)
            last_close = last_sequence[close_index]
            logger.info(f"Last known price (normalized): {last_close}")
            
            # Find the original close price from the raw data
            original_price = self._get_original_price(stock_symbol)
            logger.info(f"Last known original price: {original_price}")
            
            # Initialize variables for prediction details
            prediction_details = {
                'original_price': original_price,
                'raw_prediction': float(prediction[0, 0]),
                'scaling_method': 'metadata' if scaling_metadata else 'simple'
            }
            
            try:
                # Method 1: Use scaler with correct feature ordering
                if scaling_metadata:
                    # Use the exact feature order from training
                    features = scaling_metadata['feature_order']
                    scaler_ready = np.zeros((1, len(features)))
                    
                    # Fill in known values from the last sequence
                    for i, feature in enumerate(features):
                        if feature == "Close":
                            scaler_ready[0, i] = prediction[0, 0]
                        elif feature in FEATURES:
                            feat_idx = FEATURES.index(feature)
                            scaler_ready[0, i] = last_sequence[feat_idx]
                        else:
                            # For derived features, use their last known values
                            scaler_ready[0, i] = last_sequence[FEATURES.index(feature)] if feature in FEATURES else 0
                    
                    # Apply inverse transform
                    denormalized = scaler.inverse_transform(scaler_ready)
                    price = denormalized[0, features.index("Close")]
                else:
                    # Fallback to simpler scaling if metadata not available
                    price = self._simple_inverse_scale(prediction[0, 0], original_price, last_close)
                
                logger.info(f"Initial denormalized price: {price}")
                
                # Calculate relative change
                if original_price is not None:
                    relative_change = (price - original_price) / original_price
                    prediction_details['relative_change'] = float(relative_change)
                    prediction_details['change_percentage'] = float(relative_change * 100)
                    
                    # Check if change is large
                    if abs(relative_change) > 0.2:  # More than 20% change
                        logger.warning(f"Large price change detected: {relative_change*100:.2f}%")
                        # Calculate conservative estimate
                        conservative_price = original_price * (1 + (prediction[0, 0] - 1) * 0.1)
                        prediction_details['status'] = 'large_change_detected'
                        prediction_details['original_prediction'] = float(price)
                        prediction_details['conservative_estimate'] = float(conservative_price)
                        # Use conservative estimate as final price
                        price = conservative_price
                    else:
                        prediction_details['status'] = 'within_normal_range'
                
            except Exception as e:
                logger.warning(f"Error in scaling inverse transform: {str(e)}")
                price = self._simple_inverse_scale(prediction[0, 0], original_price, last_close)
                prediction_details['status'] = 'fallback_to_simple'
                prediction_details['error'] = str(e)
            
            # Calculate confidence score
            confidence_score = calculate_confidence_score(sequence, prediction)
            
            # Adjust confidence score based on price change
            if 'relative_change' in prediction_details:
                change_factor = abs(prediction_details['relative_change'])
                if change_factor > 0.2:
                    confidence_score *= (1 - (change_factor - 0.2))  # Reduce confidence for large changes
            
            result = {
                'prediction': float(price),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': MODEL_VERSION,
                'model_type': f'lstm_{model_type}',
                'prediction_details': prediction_details
            }
            
            # Publish to RabbitMQ
            try:
                publish_success = rabbitmq_publisher.publish_stock_quote(stock_symbol, result)
                if publish_success:
                    logger.info(f"✅ Successfully published prediction for {stock_symbol} to RabbitMQ")
                    result['rabbitmq_status'] = 'delivered'
                else:
                    logger.warning(f"⚠️ Failed to confirm RabbitMQ delivery for {stock_symbol}")
                    result['rabbitmq_status'] = 'unconfirmed'
            except Exception as e:
                logger.error(f"❌ Failed to publish to RabbitMQ: {str(e)}")
                result['rabbitmq_status'] = 'failed'
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {stock_symbol}: {str(e)}")
            raise

    def _get_original_price(self, symbol: str) -> float:
        """Get the last known original price for a symbol"""
        try:
            # First try raw data in Technology sector
            raw_file = os.path.join("data", "raw", "Technology", f"{symbol}_stock_price.csv")
            if os.path.exists(raw_file):
                df = pd.read_csv(raw_file)
                if not df.empty:
                    logger.info(f"Found price in raw data: {df['Close'].iloc[-1]}")
                    return float(df['Close'].iloc[-1])
            
            # Then try processed data
            processed_file = os.path.join("data", "processed", "specific", "Technology", f"{symbol}_processed.csv")
            if os.path.exists(processed_file):
                df = pd.read_csv(processed_file)
                if not df.empty:
                    # Get the last non-normalized close price
                    metadata_path = os.path.join("models", "specific", symbol, f"{symbol}_scaler_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            close_idx = metadata['feature_names'].index('Close')
                            min_val = metadata['min_values'][close_idx]
                            max_val = metadata['max_values'][close_idx]
                            normalized_close = df['Close'].iloc[-1]
                            original_close = normalized_close * (max_val - min_val) + min_val
                            logger.info(f"Found price in processed data: {original_close}")
                            return float(original_close)
            
            # Finally try unified data
            unified_file = os.path.join("data", "raw", "unified", f"{symbol}_stock_price.csv")
            if os.path.exists(unified_file):
                df = pd.read_csv(unified_file)
                if not df.empty:
                    logger.info(f"Found price in unified data: {df['Close'].iloc[-1]}")
                    return float(df['Close'].iloc[-1])
                    
        except Exception as e:
            logger.warning(f"Could not find original price data: {str(e)}")
            logger.exception(e)  # Log the full traceback
        return None

    def _simple_inverse_scale(self, predicted_normalized: float, original_price: float, last_close_normalized: float) -> float:
        """Simple scaling based on relative change"""
        try:
            if original_price is None:
                return predicted_normalized  # Return as is if no reference point
                
            # Calculate the relative change from the normalized values
            relative_change = (predicted_normalized - last_close_normalized) / last_close_normalized
            
            # Apply the same relative change to the original price
            result = original_price * (1 + relative_change)
            logger.info(f"Simple scaling: orig={original_price}, pred_norm={predicted_normalized}, last_norm={last_close_normalized}, change={relative_change}, result={result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in simple inverse scaling: {str(e)}")
            return predicted_normalized

    def _validate_price(self, price: float, original_price: float, predicted_normalized: float) -> float:
        """Validate and adjust the predicted price if needed"""
        percent_change = abs((price - original_price) / original_price)
        
        if percent_change > 0.1:  # More than 10% change
            logger.warning(f"Large price change detected: {percent_change*100:.2f}%")
            if percent_change > 0.2:  # More than 20% change
                # Fall back to more conservative estimate
                return original_price * (1 + (predicted_normalized - 1) * 0.1)
        
        return price

@ns_predict.route('/debug_scaling/<string:symbol>')
class ScalingDebug(Resource):
    def get(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about scaling operations for debugging"""
        try:
            # Get the latest sequence
            sequence = get_latest_sequence(symbol)
            
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
            close_index = FEATURES.index("Close")
            last_close = last_sequence[close_index]
            
            # Make a test prediction
            model = SPECIFIC_MODELS[symbol] if symbol in SPECIFIC_MODELS else GENERAL_MODEL
            prediction = model.predict(sequence)
            
            # Get original price
            original_price = self._get_original_price(symbol)
            
            # Try both scaling methods
            scaling_info['debug'] = {
                'model_type': model_type,
                'sequence_shape': sequence.shape,
                'last_known_normalized_close': float(last_close),
                'original_price': float(original_price) if original_price else None,
                'raw_prediction': float(prediction[0, 0]),
                'features_available': FEATURES,
                'scaler_feature_names': scaling_metadata['feature_names'] if 'metadata' in scaling_info else None
            }
            
            # Try metadata-based scaling
            try:
                if 'metadata' in scaling_info:
                    features = scaling_metadata['feature_order']
                    scaler_ready = np.zeros((1, len(features)))
                    
                    for i, feature in enumerate(features):
                        if feature == "Close":
                            scaler_ready[0, i] = prediction[0, 0]
                        elif feature in FEATURES:
                            feat_idx = FEATURES.index(feature)
                            scaler_ready[0, i] = last_sequence[feat_idx]
                        else:
                            scaler_ready[0, i] = last_sequence[FEATURES.index(feature)] if feature in FEATURES else 0
                    
                    denormalized = scaler.inverse_transform(scaler_ready)
                    metadata_price = denormalized[0, features.index("Close")]
                    scaling_info['debug']['metadata_based_price'] = float(metadata_price)
            except Exception as e:
                scaling_info['debug']['metadata_scaling_error'] = str(e)
            
            # Try simple scaling
            try:
                simple_price = self._simple_inverse_scale(
                    prediction[0, 0],
                    original_price,
                    last_close
                )
                scaling_info['debug']['simple_scaling_price'] = float(simple_price)
            except Exception as e:
                scaling_info['debug']['simple_scaling_error'] = str(e)
            
            # Add validation info
            if original_price:
                # Calculate relative changes
                metadata_change = None
                simple_change = None
                
                if 'metadata_based_price' in scaling_info['debug']:
                    metadata_price = scaling_info['debug']['metadata_based_price']
                    metadata_change = (metadata_price - original_price) / original_price
                    scaling_info['debug']['metadata_relative_change'] = float(metadata_change)
                    
                    # Add validation info for metadata scaling
                    if abs(metadata_change) > 0.2:
                        conservative_price = original_price * (1 + (prediction[0, 0] - 1) * 0.1)
                        scaling_info['debug']['metadata_validation'] = {
                            'status': 'large_change_detected',
                            'change_percentage': float(metadata_change * 100),
                            'conservative_estimate': float(conservative_price)
                        }
                    else:
                        scaling_info['debug']['metadata_validation'] = {
                            'status': 'within_normal_range',
                            'change_percentage': float(metadata_change * 100)
                        }
                
                if 'simple_scaling_price' in scaling_info['debug']:
                    simple_price = scaling_info['debug']['simple_scaling_price']
                    simple_change = (simple_price - original_price) / original_price
                    scaling_info['debug']['simple_relative_change'] = float(simple_change)
                    
                    # Add validation info for simple scaling
                    if abs(simple_change) > 0.2:
                        conservative_price = original_price * (1 + (prediction[0, 0] - 1) * 0.1)
                        scaling_info['debug']['simple_validation'] = {
                            'status': 'large_change_detected',
                            'change_percentage': float(simple_change * 100),
                            'conservative_estimate': float(conservative_price)
                        }
                    else:
                        scaling_info['debug']['simple_validation'] = {
                            'status': 'within_normal_range',
                            'change_percentage': float(simple_change * 100)
                        }
                
                # Add consensus information if both methods available
                if metadata_change is not None and simple_change is not None:
                    scaling_info['debug']['consensus'] = {
                        'methods_agree': abs(metadata_change - simple_change) < 0.01,
                        'average_change': float((metadata_change + simple_change) / 2 * 100),
                        'difference': float(abs(metadata_price - simple_price))
                    }
            
            return scaling_info
            
        except Exception as e:
            return {'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

def calculate_confidence_score(sequence: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate confidence score for the prediction"""
    # Example implementation - replace with your actual confidence calculation
    return 0.85

if __name__ == '__main__':
    # Check if we need to create metadata files before loading resources
    missing_metadata_count = 0
    for symbol in os.listdir("models/specific"):
        symbol_dir = os.path.join("models/specific", symbol)
        if not os.path.isdir(symbol_dir):
            continue
            
        scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
        metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
        if not os.path.exists(metadata_path) and os.path.exists(scaler_path):
            missing_metadata_count += 1
            # Try to create metadata on the fly
            logger.info(f"Creating metadata for {symbol}")
            create_metadata_for_symbol(symbol)
    
    if missing_metadata_count > 0:
        logger.warning(f"Created {missing_metadata_count} missing metadata files on startup")
    
    load_resources()
    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    finally:
        # Ensure RabbitMQ connection is closed when the server stops
        rabbitmq_publisher.close()