import os
import time
import logging
import schedule
import threading
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scheduled_tasks')

# Configuration
API_HOST = os.environ.get('API_HOST', 'localhost')
API_PORT = os.environ.get('API_PORT', '8000')
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Evaluation schedule (default: once per day)
EVALUATION_SCHEDULE = os.environ.get('EVALUATION_SCHEDULE', '00:00')  # Midnight by default

def evaluate_all_models():
    """Evaluate all models and trigger retraining if needed"""
    logger.info("Starting scheduled evaluation of all models")
    
    try:
        # Call the evaluate-all endpoint
        response = requests.get(f"{API_BASE_URL}/meta/evaluate-all?auto_retrain=true")
        
        if response.status_code == 200:
            result = response.json()
            models_evaluated = result.get('models_evaluated', 0)
            retraining_triggered = result.get('retraining_triggered', [])
            
            logger.info(f"Evaluation complete: {models_evaluated} models evaluated")
            if retraining_triggered:
                logger.info(f"Retraining triggered for {len(retraining_triggered)} models: {', '.join(retraining_triggered)}")
            else:
                logger.info("No models required retraining")
        else:
            logger.error(f"Evaluation failed with status code {response.status_code}: {response.text}")
    
    except Exception as e:
        logger.error(f"Error during scheduled evaluation: {str(e)}")

def run_scheduler():
    """Run the scheduler in a separate thread"""
    # Schedule the evaluation task
    schedule.every().day.at(EVALUATION_SCHEDULE).do(evaluate_all_models)
    
    # Also run evaluation on startup
    logger.info("Running initial model evaluation on startup")
    evaluate_all_models()
    
    logger.info(f"Scheduler started. Will evaluate models daily at {EVALUATION_SCHEDULE}")
    
    # Run the scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def start_scheduler():
    """Start the scheduler in a background thread"""
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler thread started")
    return scheduler_thread

if __name__ == "__main__":
    # This allows running the scheduler as a standalone process if needed
    logger.info("Starting scheduler as standalone process")
    run_scheduler() 