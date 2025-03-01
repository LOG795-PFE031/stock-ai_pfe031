import os
import sys
import datetime
import logging
from celery import Celery
import time
import json

# Update path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Set up celery app
app = Celery('stock_predictions',
            broker='amqp://guest:guest@localhost:5672//',
            backend='rpc://')

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'celery_tasks.process_prediction': {'queue': 'prediction_queue_1'},
        'celery_tasks.analyze_prediction': {'queue': 'prediction_queue_2'}
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.task(name='celery_tasks.process_prediction')
def process_prediction(prediction_data):
    """Process the stock prediction (simpler processor)."""
    try:
        # Log the received prediction
        ticker = prediction_data.get('ticker', 'Unknown')
        date = prediction_data.get('prediction_date', 'Unknown')
        price = prediction_data.get('predicted_open', 0.0)
        model = prediction_data.get('model', 'Unknown')
        
        logger.info(f"Consumer Queue 1 received prediction:")
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Date: {date}")
        logger.info(f"  Predicted Open: ${price:.2f}")
        logger.info(f"  Model: {model}")
        
        # Here you would typically do something with the prediction
        # For example, store it in a database, trigger a trade, etc.
        
        return {
            'status': 'processed',
            'processor': 'prediction_queue_1',
            'ticker': ticker,
            'price': price,
            'date': date
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        return {'status': 'error', 'message': str(e)}

@app.task(name='celery_tasks.analyze_prediction')
def analyze_prediction(prediction_data):
    """Process the stock prediction with additional analysis (advanced processor)."""
    try:
        # Log the received prediction
        ticker = prediction_data.get('ticker', 'Unknown')
        date = prediction_data.get('prediction_date', 'Unknown')
        price = prediction_data.get('predicted_open', 0.0)
        model = prediction_data.get('model', 'Unknown')
        
        # More detailed logging in this consumer
        logger.info("=" * 50)
        logger.info(f"ALERT: New stock prediction received from {model} model")
        logger.info(f"Stock: {ticker}")
        logger.info(f"Date: {date}")
        logger.info(f"Predicted Opening Price: ${price:.2f}")
        
        # Simulate doing some additional processing
        logger.info("Performing additional analysis...")
        time.sleep(2)  # Simulate complex work
        
        # Calculate a price range
        price_low = price * 0.95
        price_high = price * 1.05
        logger.info(f"Predicted price range: ${price_low:.2f} - ${price_high:.2f}")
        
        # Calculate confidence score (just a demo)
        confidence = 0.85
        logger.info(f"Prediction confidence: {confidence:.2%}")
        logger.info("=" * 50)
        
        return {
            'status': 'analyzed',
            'processor': 'prediction_queue_2',
            'ticker': ticker,
            'price': price,
            'date': date,
            'price_range': {
                'low': price_low,
                'high': price_high
            },
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error analyzing prediction: {e}")
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    app.start()