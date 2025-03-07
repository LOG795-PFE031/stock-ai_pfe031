"""
S&P 500 Stock Prediction Model Training for Google Colab

This script is designed to be run in Google Colab to train stock prediction models.
It uses GPU-optimized implementations for faster training and better memory management.
"""

import sys
import os
import time
import subprocess
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('colab_train')

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Add the script directory to the path
sys.path.append('data/script')

# Performance optimization settings
FAST_MODE = True  # Set to True to use faster data preparation (smaller sample, parallel processing)
MAX_SYMBOLS = 10  # Maximum number of symbols to process in fast mode (None for all)
BATCH_SIZE = 128  # Batch size for training (larger for GPU)
SEQUENCE_LENGTH = 60  # Number of time steps to use for sequence data
EPOCHS = 50  # Maximum number of epochs for training

# GPU/CPU settings
FORCE_CPU_LSTM = True  # Set to True to use CPU for LSTM operations and GPU for other operations

# Check for required packages without reinstalling
logger.info("Checking for required packages...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import psutil
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    logger.info("All required packages are already installed.")
except ImportError as e:
    logger.warning(f"Missing package: {e}")
    logger.info("Please install missing packages manually to avoid environment conflicts.")

# Force TensorFlow to see the GPU - Enhanced configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Fix for TensorFlow GPU detection
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "4"
os.environ["TF_USE_CUDNN_AUTOTUNE"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"

# Additional environment variables to help with GPU detection
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce TensorFlow logging verbosity
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Import TensorFlow after setting environment variables
try:
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Set memory growth before any other TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Memory growth enabled for {device}")
            except Exception as e:
                logger.error(f"Error setting memory growth: {e}")
except ImportError:
    logger.error("Failed to import TensorFlow. Please check your installation.")
    tf = None

def check_gpu():
    """Check if GPU is available and print device information"""
    logger.info("Checking for GPU availability...")
    
    if tf is None:
        logger.error("TensorFlow is not available. Cannot check GPU.")
        return False
    
    # Check TensorFlow GPU detection
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"TensorFlow detected GPU: {len(gpus)} GPU(s) found")
        for gpu in gpus:
            logger.info(f"  {gpu}")
        
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            logger.info(f"GPU Details: {gpu_details}")
        except Exception as e:
            logger.warning(f"Could not get GPU details: {e}")
        
        # Try to get memory info using nvidia-smi
        try:
            logger.info("\nGPU Memory Information:")
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            logger.info(result.stdout.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not get detailed GPU memory information: {e}")
            
        # Verify GPU is actually usable by TensorFlow
        logger.info("\nVerifying GPU is usable by TensorFlow...")
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                # Force execution
                result = c.numpy().mean()
                logger.info(f"GPU test successful! Matrix multiplication result: {result}")
                return True
        except RuntimeError as e:
            logger.error(f"ERROR: Could not use GPU for computation: {e}")
            logger.info("Will try to fix GPU visibility...")
            
            # Try to fix GPU visibility
            try:
                # Clear TensorFlow session
                tf.keras.backend.clear_session()
                logger.info("Cleared TensorFlow session")
                
                # Try again with explicit device placement
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                    # Force execution
                    result = c.numpy().mean()
                    logger.info(f"GPU test successful after fix! Matrix multiplication result: {result}")
                    return True
            except RuntimeError as e2:
                logger.error(f"ERROR: Could not use GPU even after fix: {e2}")
    else:
        logger.warning("WARNING: No GPU detected by TensorFlow.")
        
        # Check if nvidia-smi can detect the GPU
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            if "NVIDIA-SMI" in output:
                logger.info("\nNVIDIA GPU detected by system but not by TensorFlow!")
                logger.info(output)
                logger.info("\nTrying to fix TensorFlow GPU detection...")
                
                # Try to manually load CUDA libraries
                logger.info("Attempting to manually load CUDA libraries...")
                try:
                    # Try to load CUDA runtime
                    from ctypes import cdll
                    cdll.LoadLibrary("libcudart.so")
                    logger.info("Successfully loaded libcudart.so")
                except Exception as e:
                    logger.warning(f"Could not manually load CUDA runtime: {e}")
                
                # Clear TensorFlow session
                if tf is not None:
                    tf.keras.backend.clear_session()
                    logger.info("Cleared TensorFlow session")
                
                # Check again
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Success! TensorFlow now sees {len(gpus)} GPU(s)")
                    return True
                else:
                    logger.warning("Still no GPU detected by TensorFlow after fix attempt.")
                    
                    # Last resort: Try to use mixed precision
                    try:
                        logger.info("Trying mixed precision as a last resort...")
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info(f"Set mixed precision policy to {policy.name}")
                        return True
                    except Exception as e:
                        logger.warning(f"Could not set mixed precision policy: {e}")
            else:
                logger.warning("No NVIDIA GPU detected by system.")
        except Exception as e:
            logger.warning(f"Could not run nvidia-smi to check for GPU: {e}")
    
    logger.info("\nTensorFlow version: %s", tf.__version__ if tf else "Not available")
    logger.info("Eager execution enabled: %s", tf.executing_eagerly() if tf else "Not available")
    logger.info("Mixed precision policy: %s", tf.keras.mixed_precision.global_policy().name if tf else "Not available")
    
    # Check if CUDA is available
    logger.info("CUDA available: %s", tf.test.is_built_with_cuda() if tf else "Not available")
    
    # Check if we're using hybrid CPU/GPU mode
    logger.info("Using hybrid CPU/GPU mode (LSTM on CPU, other operations on GPU): %s", FORCE_CPU_LSTM)
    
    return False

def fix_csv_parsing_error():
    """Fix CSV parsing error by reading with Python engine"""
    logger.info("Attempting to fix CSV parsing error...")
    
    unified_file = os.path.join("data/processed/general", "all_stocks_processed.csv")
    if not os.path.exists(unified_file):
        logger.error(f"ERROR: Unified dataset not found at {unified_file}")
        return False
    
    try:
        # Try reading with Python engine
        logger.info(f"Reading {unified_file} with Python engine...")
        data = pd.read_csv(unified_file, engine='python')
        
        # Save back with proper formatting
        fixed_file = os.path.join("data/processed/general", "all_stocks_processed_fixed.csv")
        data.to_csv(fixed_file, index=False)
        logger.info(f"Fixed CSV saved to {fixed_file}")
        
        # Replace original with fixed version
        os.rename(fixed_file, unified_file)
        logger.info(f"Original CSV replaced with fixed version")
        return True
    except Exception as e:
        logger.error(f"ERROR fixing CSV: {e}")
        return False

# Training configuration - modify these variables as needed
TRAIN_GENERAL = True  # Set to True to train the general model
TRAIN_SPECIFIC = True  # Set to True to train specific models
EVALUATE = True  # Set to True to evaluate models
USE_TRANSFER_LEARNING = True  # Set to True to use transfer learning for specific models
SYMBOLS = None  # Set to a list of symbols to train specific ones, e.g., ['AAPL', 'MSFT', 'GOOGL']
SAMPLE_FRACTION = 0.2  # Fraction of data to use for general model (0.2 = 20%)

def main():
    """Main function to train and evaluate models"""
    start_time = time.time()
    logger.info(f"Starting GPU-optimized S&P 500 stock prediction model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Fix CSV parsing error
    csv_fixed = fix_csv_parsing_error()
    if not csv_fixed:
        logger.warning("WARNING: Could not fix CSV parsing error. Training may fail.")
    
    # Apply fast mode settings if enabled
    sample_fraction = SAMPLE_FRACTION
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'] if FAST_MODE and SYMBOLS is None else SYMBOLS
    
    if FAST_MODE:
        logger.info("\nFAST MODE ENABLED: Using reduced dataset and limited symbols for quicker training")
        logger.info(f"Sample fraction: {sample_fraction}, Symbols: {symbols if symbols else 'First ' + str(MAX_SYMBOLS) + ' symbols'}")
    
    # Create trainer instance
    try:
        if gpu_available:
            logger.info("\nUsing GPU-optimized trainer")
            # Import the GPU-optimized trainer
            from gpu_optimized_trainer import GPUOptimizedTrainer
            trainer = GPUOptimizedTrainer(
                processed_dir="data/processed",
                models_dir="models",
                sequence_length=SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
            )
        else:
            logger.info("\nUsing CPU trainer (GPU not available)")
            # Import the standard trainer
            from train_sp500_models import SP500ModelTrainer
            trainer = SP500ModelTrainer(
                processed_dir="data/processed",
                models_dir="models",
                sequence_length=SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
            )
        
        # Train general model
        if TRAIN_GENERAL:
            logger.info("\n=== TRAINING GENERAL MODEL ===")
            try:
                if gpu_available and FAST_MODE:
                    # Use the fast implementation with our updated parameter names
                    trainer._train_general_model_fast(sample_fraction=sample_fraction, max_symbols=symbols)
                else:
                    # Use the standard implementation
                    trainer.train_general_model(sample_fraction=sample_fraction)
            except Exception as e:
                logger.error(f"\nERROR during general model training: {str(e)}")
                logger.info("Falling back to CPU training...")
                
                # Fallback to CPU training
                from train_sp500_models import SP500ModelTrainer
                cpu_trainer = SP500ModelTrainer(
                    processed_dir="data/processed",
                    models_dir="models",
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=64,  # Smaller batch size for CPU
                    epochs=EPOCHS
                )
                cpu_trainer.train_general_model(sample_fraction=sample_fraction)
        
        # Train specific models
        if TRAIN_SPECIFIC:
            logger.info("\n=== TRAINING SPECIFIC MODELS ===")
            if FAST_MODE and symbols is None:
                # Get first MAX_SYMBOLS symbols
                all_symbols = trainer.get_available_symbols()
                symbols = all_symbols[:MAX_SYMBOLS] if MAX_SYMBOLS else all_symbols
                logger.info(f"Fast mode: Training on {len(symbols)} symbols: {symbols}")
            
            try:
                trainer.train_all_models(symbols=symbols, use_transfer_learning=USE_TRANSFER_LEARNING)
            except Exception as e:
                logger.error(f"\nERROR during specific model training: {str(e)}")
                logger.info("Falling back to CPU training for specific models...")
                
                # Fallback to CPU training
                from train_sp500_models import SP500ModelTrainer
                cpu_trainer = SP500ModelTrainer(
                    processed_dir="data/processed",
                    models_dir="models",
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=64,  # Smaller batch size for CPU
                    epochs=EPOCHS
                )
                cpu_trainer.train_all_models(symbols=symbols, use_transfer_learning=USE_TRANSFER_LEARNING)
        
        # Evaluate models
        if EVALUATE:
            logger.info("\n=== EVALUATING MODELS ===")
            try:
                results = trainer.evaluate_all_models(symbols=symbols)
                logger.info("\nEvaluation results summary:")
                if not results.empty:
                    logger.info(results.describe())
                else:
                    logger.warning("No evaluation results available - empty DataFrame returned")
                
                # Save results to CSV
                results_path = os.path.join("models", "evaluation_results.csv")
                results.to_csv(results_path)
                logger.info(f"Full evaluation results saved to {results_path}")
            except Exception as e:
                logger.error(f"\nERROR during model evaluation: {str(e)}")
                logger.info("Falling back to CPU evaluation...")
                
                # Fallback to CPU evaluation
                from train_sp500_models import SP500ModelTrainer
                cpu_trainer = SP500ModelTrainer(
                    processed_dir="data/processed",
                    models_dir="models",
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=64,
                    epochs=EPOCHS
                )
                results = cpu_trainer.evaluate_all_models(symbols=symbols)
                logger.info("\nEvaluation results summary:")
                if not results.empty:
                    logger.info(results.describe())
                else:
                    logger.warning("No evaluation results available - empty DataFrame returned")
                
                # Save results to CSV
                results_path = os.path.join("models", "evaluation_results.csv")
                results.to_csv(results_path)
                logger.info(f"Full evaluation results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: {str(e)}")
        logger.info("Falling back to standard CPU implementation...")
        
        # Complete fallback to CPU
        from train_sp500_models import SP500ModelTrainer
        cpu_trainer = SP500ModelTrainer(
            processed_dir="data/processed",
            models_dir="models",
            sequence_length=SEQUENCE_LENGTH,
            batch_size=64,  # Smaller batch size for CPU
            epochs=EPOCHS
        )
        
        if TRAIN_GENERAL:
            logger.info("\n=== TRAINING GENERAL MODEL (CPU FALLBACK) ===")
            cpu_trainer.train_general_model(sample_fraction=sample_fraction)
        
        if TRAIN_SPECIFIC:
            logger.info("\n=== TRAINING SPECIFIC MODELS (CPU FALLBACK) ===")
            if symbols is None and FAST_MODE:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:MAX_SYMBOLS]
            cpu_trainer.train_all_models(symbols=symbols, use_transfer_learning=USE_TRANSFER_LEARNING)
        
        if EVALUATE:
            logger.info("\n=== EVALUATING MODELS (CPU FALLBACK) ===")
            results = cpu_trainer.evaluate_all_models(symbols=symbols)
            logger.info("\nEvaluation results summary:")
            if not results.empty:
                logger.info(results.describe())
            else:
                logger.warning("No evaluation results available - empty DataFrame returned")
            results_path = os.path.join("models", "evaluation_results.csv")
            results.to_csv(results_path)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logger.error(f"ERROR: An exception occurred during training: {str(e)}")
        traceback.print_exc()