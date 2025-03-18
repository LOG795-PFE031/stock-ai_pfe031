"""
API endpoints for model training operations.
"""
from flask_restx import Resource, Namespace, fields, Api
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
import glob

from ..training.trainer import ModelTrainer
from ..core.config import Config

# Create API instance
api = Api()

# Create namespace for training operations
ns = Namespace('training', description='Model training operations')

# Define request/response models
training_request_model = ns.model('TrainingRequest', {
    'symbols': fields.List(fields.String, description='List of stock symbols to train models for'),
    'model_types': fields.List(fields.String, description='List of model types to train (lstm, prophet)'),
    'force_retrain': fields.Boolean(description='Whether to force retraining of existing models', default=False)
})

training_response_model = ns.model('TrainingResponse', {
    'status': fields.String(description='Training status (success/failed)'),
    'message': fields.String(description='Status message'),
    'results': fields.Raw(description='Training results for each symbol and model type'),
    'start_time': fields.String(description='Training start time (ISO format)'),
    'end_time': fields.String(description='Training end time (ISO format)')
})

@ns.route('/train')
class ModelTraining(Resource):
    @ns.doc('train_models',
             description='Train models for specified symbols and types',
             responses={
                 200: ('Training started successfully', training_response_model),
                 400: 'Invalid request parameters',
                 500: 'Training error'
             })
    @ns.expect(training_request_model)
    @ns.marshal_with(training_response_model)
    def post(self) -> Dict:
        """Train models for specified symbols and model types"""
        try:
            # Get request data
            data = request.get_json()
            symbols = data.get('symbols')
            model_types = data.get('model_types')
            force_retrain = data.get('force_retrain', False)
            
            # Validate request
            if not symbols or not model_types:
                return {
                    'status': 'error',
                    'message': 'Both symbols and model_types are required',
                    'results': {},
                    'start_time': None,
                    'end_time': None
                }, 400
            
            # Initialize trainer
            config = Config()
            trainer = ModelTrainer(config)
            
            # Start training
            start_time = datetime.now()
            results = trainer.train_all_models(symbols=symbols, model_types=model_types)
            end_time = datetime.now()
            
            # Prepare response
            response = {
                'status': 'success',
                'message': f'Training completed for {len(symbols)} symbols',
                'results': results,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            return response, 200
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Training failed: {str(e)}',
                'results': {},
                'start_time': None,
                'end_time': None
            }, 500

@ns.route('/status')
class TrainingStatus(Resource):
    @ns.doc('get_training_status',
             description='Get status of recent training operations',
             responses={
                 200: 'Training status retrieved successfully',
                 500: 'Error retrieving status'
             })
    def get(self) -> Dict:
        """Get status of recent training operations"""
        try:
            # Get logs directory from config
            config = Config()
            logs_dir = config.data.LOGS_DIR
            
            # Find most recent training log
            log_files = glob.glob(os.path.join(logs_dir, 'training_*.log'))
            if not log_files:
                return {
                    'status': 'no_training_history',
                    'message': 'No training history found',
                    'last_training': None
                }, 200
            
            latest_log = max(log_files, key=os.path.getctime)
            
            # Read last few lines of log
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-10:]  # Last 10 lines
            
            return {
                'status': 'success',
                'message': 'Training status retrieved successfully',
                'last_training': {
                    'log_file': latest_log,
                    'last_lines': lines,
                    'timestamp': datetime.fromtimestamp(os.path.getctime(latest_log)).isoformat()
                }
            }, 200
            
        except Exception as e:
            logging.error(f"Error getting training status: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to get training status: {str(e)}',
                'last_training': None
            }, 500 