"""
NASDAQ-100 Stock Prediction Model Training for Google Colab

This script is adapted from the S&P 500 version to train NASDAQ-100 stock prediction models.
"""

# Keep all imports and GPU setup code the same as colab_train.py
# Only changing the relevant configuration and paths

# Performance optimization settings
FAST_MODE = True
MAX_SYMBOLS = 10  # Maximum number of symbols to process in fast mode
BATCH_SIZE = 128  # Batch size for training
SEQUENCE_LENGTH = 60  # Number of time steps
EPOCHS = 50

# Default symbols for NASDAQ-100 fast mode
DEFAULT_NASDAQ_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CSCO', 'INTC']

def main():
    """Main function to train and evaluate NASDAQ models"""
    start_time = time.time()
    logger.info(f"Starting GPU-optimized NASDAQ-100 stock prediction model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Fix CSV parsing error
    csv_fixed = fix_csv_parsing_error()
    if not csv_fixed:
        logger.warning("WARNING: Could not fix CSV parsing error. Training may fail.")
    
    # Apply fast mode settings if enabled
    sample_fraction = SAMPLE_FRACTION
    symbols = DEFAULT_NASDAQ_SYMBOLS if FAST_MODE and SYMBOLS is None else SYMBOLS
    
    if FAST_MODE:
        logger.info("\nFAST MODE ENABLED: Using reduced dataset and limited NASDAQ symbols")
        logger.info(f"Sample fraction: {sample_fraction}, Symbols: {symbols if symbols else 'First ' + str(MAX_SYMBOLS) + ' symbols'}")
    
    # Create trainer instance
    try:
        if gpu_available:
            logger.info("\nUsing GPU-optimized trainer")
            from training.GPUOptimizedTrainerNASDAQ import GPUOptimizedTrainerNASDAQ
            trainer = GPUOptimizedTrainerNASDAQ(
                processed_dir="data/processed",
                models_dir="models",
                sequence_length=SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
            )
        else:
            logger.info("\nUsing CPU trainer (GPU not available)")
            from train_nasdaq_models import NASDAQModelTrainer
            trainer = NASDAQModelTrainer(
                processed_dir="data/processed",
                models_dir="models",
                sequence_length=SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
            )
        
        # Rest of the training code remains the same as colab_train.py
        # Just update the logging messages to reference NASDAQ instead of S&P 500

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logger.error(f"ERROR: An exception occurred during training: {str(e)}")
        traceback.print_exc() 