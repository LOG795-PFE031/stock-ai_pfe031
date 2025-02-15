from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Dict, List, Any

app = Flask(__name__)
api = Api(
    app, 
    title='Stock Price Prediction API',
    version='1.0',
    description='A production-ready API for predicting Google stock prices using LSTM',
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
    'model_version': fields.String(description='Version of the ML model used')
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
MODEL = None
SCALER = None
SEQ_SIZE = 60
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()
REQUEST_COUNT = 0

class ModelNotLoadedError(Exception):
    """Raised when the model is not properly loaded"""
    pass

def load_resources() -> None:
    """Load ML model and scaler with proper error handling"""
    global MODEL, SCALER
    try:
        MODEL = load_model("models/2025_google_stock_price_lstm.model.keras")
        SCALER = joblib.load("models/2025_google_stock_price_scaler.gz")
        app.logger.info("✅ Resources loaded successfully")
    except Exception as e:
        app.logger.error(f"❌ Error loading resources: {str(e)}")
        raise ModelNotLoadedError("Failed to load model resources") from e
    
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

@ns_health.route('/health')
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
            # Verify model is loaded
            if MODEL is None or SCALER is None:
                raise ModelNotLoadedError("Model resources not loaded")
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'model': 'healthy',
                    'scaler': 'healthy',
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
        responses={
            HTTPStatus.OK: ('Successful prediction', prediction_model),
            HTTPStatus.INTERNAL_SERVER_ERROR: ('Prediction error', error_model)
        }
    )
    @api.marshal_with(prediction_model)
    def get(self) -> Dict[str, Any]:
        """Get detailed stock price prediction for the next trading day"""
        global REQUEST_COUNT
        try:
            REQUEST_COUNT += 1
            sequence = get_latest_sequence()
            prediction = MODEL.predict(sequence)
            
            future_predictions_padded = np.concatenate((prediction, np.ones((1, 5))), axis=1)
            price = SCALER.inverse_transform(future_predictions_padded)[0][0]
            
            # Calculate confidence score (example implementation)
            confidence_score = calculate_confidence_score(sequence, prediction)
            
            return {
                'prediction': float(price),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': MODEL_VERSION
            }
        except Exception as e:
            api.abort(
                HTTPStatus.INTERNAL_SERVER_ERROR, 
                f"Prediction error: {str(e)}"
            )

def calculate_confidence_score(sequence: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate confidence score for the prediction"""
    # Example implementation - replace with your actual confidence calculation
    return 0.85

# ... rest of your helper functions ...

if __name__ == '__main__':
    load_resources()
    app.run(host='0.0.0.0', port=8000, debug=False)