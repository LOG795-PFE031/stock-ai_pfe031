import os
import json
import logging
import pika
import time
import sys
import importlib
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/training_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('training_worker')

# RabbitMQ connection parameters
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Maximum number of retries for connecting to RabbitMQ
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds

def connect_to_rabbitmq() -> Optional[pika.BlockingConnection]:
    """Connect to RabbitMQ with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT} (attempt {attempt+1}/{MAX_RETRIES})")
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=600,  # 10 minutes heartbeat
                    blocked_connection_timeout=300  # 5 minutes timeout
                )
            )
            logger.info("Successfully connected to RabbitMQ")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Could not connect to RabbitMQ.")
                return None

def preprocess_data(symbol: str, params: Dict[str, Any]) -> bool:
    """Run data preprocessing for a specific symbol"""
    try:
        logger.info(f"Starting preprocessing for {symbol}")
        
        # Import the preprocessing module
        sys.path.append('/app/data/script')
        from data.script.preprocess_sp500_data import SP500DataPreprocessor
        
        # Create preprocessor instance
        preprocessor = SP500DataPreprocessor(
            raw_dir=params.get('raw_dir', "data/raw"),
            processed_dir=params.get('processed_dir', "data/processed"),
            models_dir=params.get('models_dir', MODELS_DIR)
        )
        
        # If symbol is specified, preprocess just that symbol
        if symbol and symbol.lower() != 'all':
            # Find the stock's data file
            stock_files = []
            for sector_dir in os.listdir(os.path.join(preprocessor.raw_dir)):
                sector_path = os.path.join(preprocessor.raw_dir, sector_dir)
                if os.path.isdir(sector_path):
                    stock_file = os.path.join(sector_path, f"{symbol}_stock_price.csv")
                    if os.path.exists(stock_file):
                        stock_files.append(stock_file)
            
            if not stock_files:
                logger.error(f"No data file found for {symbol}")
                return False
            
            # Process the specific stock
            processed_df, symbol, sector = preprocessor.preprocess_stock(stock_files[0])
            if processed_df is not None:
                # Save individual stock data
                safe_sector = str(sector).replace('/', '_')
                output_dir = os.path.join(preprocessor.processed_dir, "specific", safe_sector)
                os.makedirs(output_dir, exist_ok=True)
                
                if symbol is not None:
                    output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
                    processed_df.to_csv(output_file, index=False)
                    logger.info(f"Successfully processed {symbol}")
                    return True
            
            logger.error(f"Failed to process {symbol}")
            return False
        else:
            # Process all stocks
            preprocessor.preprocess_all_stocks()
            logger.info("Successfully processed all stocks")
            return True
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def train_general_model(params: Dict[str, Any]) -> bool:
    """Train the general model for all stocks"""
    try:
        logger.info("Starting general model training")
        
        # Import the training module
        sys.path.append('/app/data/script')
        from data.script.train_sp500_models import SP500ModelTrainer
        
        # Create trainer instance
        trainer = SP500ModelTrainer(
            processed_dir=params.get('processed_dir', "data/processed"),
            models_dir=params.get('models_dir', MODELS_DIR),
            sequence_length=params.get('sequence_length', 60),
            batch_size=params.get('batch_size', 32),
            epochs=params.get('epochs', 50)
        )
        
        # Train the general model
        sample_fraction = params.get('sample_fraction', 0.1)  # Default to 10% of data for faster training
        trainer.train_general_model(sample_fraction=sample_fraction)
        
        logger.info("General model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during general model training: {str(e)}")
        return False

def train_specific_model(symbol: str, params: Dict[str, Any]) -> bool:
    """Train a stock-specific model"""
    try:
        logger.info(f"Starting specific model training for {symbol}")
        
        # Import the training module
        sys.path.append('/app/data/script')
        from data.script.train_sp500_models import SP500ModelTrainer
        
        # Create trainer instance
        trainer = SP500ModelTrainer(
            processed_dir=params.get('processed_dir', "data/processed"),
            models_dir=params.get('models_dir', MODELS_DIR),
            sequence_length=params.get('sequence_length', 60),
            batch_size=params.get('batch_size', 32),
            epochs=params.get('epochs', 50)
        )
        
        # Train the specific model
        use_transfer_learning = params.get('use_transfer_learning', True)
        trainer.train_specific_model(symbol, use_transfer_learning=use_transfer_learning)
        
        logger.info(f"Specific model training for {symbol} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during specific model training for {symbol}: {str(e)}")
        return False

def evaluate_model(symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a model and return metrics"""
    try:
        logger.info(f"Starting model evaluation for {symbol}")
        
        # Import the training module
        sys.path.append('/app/data/script')
        from data.script.train_sp500_models import SP500ModelTrainer
        
        # Create trainer instance
        trainer = SP500ModelTrainer(
            processed_dir=params.get('processed_dir', "data/processed"),
            models_dir=params.get('models_dir', MODELS_DIR)
        )
        
        # Evaluate the model
        use_general = params.get('use_general', False)
        metrics = trainer.evaluate_model(symbol, use_general=use_general)
        
        logger.info(f"Model evaluation for {symbol} completed successfully")
        logger.info(f"Metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation for {symbol}: {str(e)}")
        return {"error": str(e)}

def download_data(symbol: str, params: Dict[str, Any]) -> bool:
    """Download stock data for a specific symbol or all S&P 500 stocks"""
    try:
        logger.info(f"Starting data download for {'all stocks' if not symbol or symbol.lower() == 'all' else symbol}")
        
        # Import the download module
        sys.path.append('/app/data/script')
        from data.script.download_sp500_data import SP500DataDownloader
        
        # Create downloader instance
        downloader = SP500DataDownloader(
            output_dir=params.get('output_dir', "data/raw"),
            max_workers=params.get('max_workers', 5),
            base_delay=params.get('base_delay', 1)
        )
        
        # Download data
        if not symbol or symbol.lower() == 'all':
            # Download all stocks
            downloader.download_all_stocks(batch_size=params.get('batch_size', 10))
        else:
            # Download specific stock
            symbols = downloader.get_sp500_symbols()
            for sym, sector in symbols:
                if sym.upper() == symbol.upper():
                    downloader.download_stock_data((sym, sector))
                    logger.info(f"Downloaded data for {sym}")
                    return True
            
            logger.error(f"Symbol {symbol} not found in S&P 500")
            return False
        
        logger.info("Data download completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during data download: {str(e)}")
        return False

def process_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a training job based on its type"""
    job_id = job_data.get('job_id', 'unknown')
    job_type = job_data.get('job_type', 'unknown')
    symbol = job_data.get('symbol', '')
    params = job_data.get('params', {})
    
    logger.info(f"Processing job {job_id} of type {job_type} for symbol {symbol}")
    
    result = {
        'job_id': job_id,
        'job_type': job_type,
        'symbol': symbol,
        'start_time': datetime.now().isoformat(),
        'status': 'failed',  # Default to failed, will be updated if successful
        'error': None
    }
    
    try:
        # Process based on job type
        if job_type == 'download':
            success = download_data(symbol, params)
            result['status'] = 'completed' if success else 'failed'
            if not success:
                result['error'] = 'Failed to download data'
                
        elif job_type == 'preprocess':
            success = preprocess_data(symbol, params)
            result['status'] = 'completed' if success else 'failed'
            if not success:
                result['error'] = 'Failed to preprocess data'
                
        elif job_type == 'train_general':
            success = train_general_model(params)
            result['status'] = 'completed' if success else 'failed'
            if not success:
                result['error'] = 'Failed to train general model'
                
        elif job_type == 'train_specific':
            success = train_specific_model(symbol, params)
            result['status'] = 'completed' if success else 'failed'
            if not success:
                result['error'] = 'Failed to train specific model'
                
        elif job_type == 'evaluate':
            metrics = evaluate_model(symbol, params)
            result['metrics'] = metrics
            result['status'] = 'completed' if 'error' not in metrics else 'failed'
            if 'error' in metrics:
                result['error'] = metrics['error']
                
        else:
            logger.error(f"Unknown job type: {job_type}")
            result['error'] = f"Unknown job type: {job_type}"
            
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        result['error'] = str(e)
        
    result['end_time'] = datetime.now().isoformat()
    logger.info(f"Job {job_id} completed with status: {result['status']}")
    
    return result

def callback(ch, method, properties, body):
    """Callback function for processing RabbitMQ messages"""
    try:
        # Parse job data
        job_data = json.loads(body)
        job_id = job_data.get('job_id', 'unknown')
        
        logger.info(f"Received job {job_id}")
        
        # Process the job
        result = process_job(job_data)
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
        logger.info(f"Job {job_id} acknowledged")
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        # Reject the message and requeue it
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def main():
    """Main function to start the worker"""
    logger.info("Starting training worker")
    
    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "general"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "specific"), exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Connect to RabbitMQ
    connection = None
    while connection is None:
        connection = connect_to_rabbitmq()
        if connection is None:
            logger.info("Waiting before trying to reconnect...")
            time.sleep(10)
    
    channel = connection.channel()
    
    # Declare the queue
    channel.queue_declare(queue='training_jobs', durable=True)
    
    # Set prefetch count to 1 to ensure fair dispatch
    channel.basic_qos(prefetch_count=1)
    
    # Register the callback
    channel.basic_consume(queue='training_jobs', on_message_callback=callback)
    
    logger.info("Waiting for training jobs. To exit press CTRL+C")
    
    try:
        # Start consuming messages
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
        channel.stop_consuming()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if connection and connection.is_open:
            connection.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    main() 