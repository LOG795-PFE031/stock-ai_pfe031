"""
GPU-Optimized S&P 500 Stock Prediction Model Trainer

This module extends the SP500ModelTrainer class with GPU optimizations
for faster training on Google Colab or other GPU-enabled environments.
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from typing import List, Optional, Dict, Any, Tuple

# Import the original trainer class
from data.script.train_sp500_models import SP500ModelTrainer

class GPUOptimizedTrainer(SP500ModelTrainer):
    """
    GPU-optimized version of the SP500ModelTrainer class
    with additional features for GPU utilization monitoring and performance.
    """
    
    def __init__(self, processed_dir="data/processed", models_dir="models", 
                 sequence_length=60, batch_size=128, epochs=50):
        """
        Initialize the GPU-optimized trainer with larger batch size
        
        Args:
            processed_dir: Directory containing processed data
            models_dir: Directory to save trained models
            sequence_length: Number of time steps to use for sequence data
            batch_size: Batch size for training (increased for GPU)
            epochs: Maximum number of epochs for training
        """
        # Call the parent class constructor
        super().__init__(
            processed_dir=processed_dir,
            models_dir=models_dir,
            sequence_length=sequence_length,
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Set up GPU
        self.has_gpu = self.setup_gpu()
        
        # Create GPU monitoring callback
        self.gpu_monitor = self.GPUMonitorCallback(log_interval=2)
    
    def setup_gpu(self) -> bool:
        """Configure GPU settings for optimal performance"""
        self.logger.info("Setting up GPU...")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            self.logger.warning("No GPU found. Training will run on CPU which will be much slower.")
            return False
        
        self.logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        # Configure memory growth to avoid allocating all memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info("Memory growth enabled for GPUs")
        except RuntimeError as e:
            self.logger.warning(f"Error setting memory growth: {e}")
        
        # Enable mixed precision training for faster computation
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        self.logger.info(f"Mixed precision policy set to: {policy.name}")
        
        # Set TensorFlow to use the GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        # Additional performance optimizations
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        self.logger.info("XLA JIT compilation enabled")
        
        # Print GPU info
        self.logger.info("\nGPU Information:")
        os.system('nvidia-smi')
        
        return True
    
    class GPUMonitorCallback(tf.keras.callbacks.Callback):
        """Callback to monitor GPU usage during training"""
        def __init__(self, log_interval=5):
            super().__init__()
            self.log_interval = log_interval
            self.start_time = None
        
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print("Starting training with GPU monitoring...")
        
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1} starting...")
        
        def on_epoch_end(self, epoch, logs=None):
            # Log GPU stats every few epochs
            if (epoch + 1) % self.log_interval == 0:
                # Print GPU stats
                print("\nGPU Statistics:")
                os.system('nvidia-smi')
                
                # Calculate time elapsed
                elapsed = time.time() - self.start_time
                print(f"Time elapsed: {elapsed:.2f} seconds")
                
                # Print metrics
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                print(f"Training metrics: {metrics_str}")
    
    def train_general_model(self, sample_fraction=1.0) -> tf.keras.Model:
        """
        Train the general model on all stocks with GPU optimizations
        
        Args:
            sample_fraction: Fraction of data to use (0.1 = 10%)
            
        Returns:
            Trained Keras model
        """
        # Add GPU monitoring callback if GPU is available
        callbacks = []
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Call the parent class method with our callbacks
        return super().train_general_model(sample_fraction=sample_fraction, additional_callbacks=callbacks)
    
    def train_specific_model(self, symbol: str, use_transfer_learning: bool = True) -> Optional[tf.keras.Model]:
        """
        Train a stock-specific model with GPU optimizations
        
        Args:
            symbol: Stock symbol
            use_transfer_learning: Whether to use transfer learning from the general model
            
        Returns:
            Trained Keras model
        """
        # Add GPU monitoring callback if GPU is available
        callbacks = []
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Call the parent class method with our callbacks
        return super().train_specific_model(symbol, use_transfer_learning, additional_callbacks=callbacks)
    
    def train_all_models(self, symbols: Optional[List[str]] = None, use_transfer_learning: bool = True):
        """
        Train all models (general and specific) with GPU optimizations
        
        Args:
            symbols: List of symbols to train specific models for (None for all)
            use_transfer_learning: Whether to use transfer learning for specific models
        """
        # Add GPU monitoring callback if GPU is available
        callbacks = []
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Call the parent class method with our callbacks
        return super().train_all_models(symbols, use_transfer_learning, additional_callbacks=callbacks)

# Main function for direct execution
def main():
    """Main function to train and evaluate models with GPU optimizations"""
    print("Starting GPU-optimized S&P 500 stock prediction model training...")
    
    # Create trainer instance
    trainer = GPUOptimizedTrainer(
        processed_dir="data/processed",
        models_dir="models",
        sequence_length=60,
        batch_size=128,  # Larger batch size for GPU
        epochs=50
    )
    
    # Training configuration - modify these variables as needed
    TRAIN_GENERAL = True  # Set to True to train the general model
    TRAIN_SPECIFIC = True  # Set to True to train specific models
    EVALUATE = True  # Set to True to evaluate models
    USE_TRANSFER_LEARNING = True  # Set to True to use transfer learning for specific models
    SYMBOLS = None  # Set to a list of symbols to train specific ones, e.g., ['AAPL', 'MSFT', 'GOOGL']
    SAMPLE_FRACTION = 1.0  # Fraction of data to use for general model (0.1 = 10%)
    
    # Train general model
    if TRAIN_GENERAL:
        print("Training general model...")
        trainer.train_general_model(sample_fraction=SAMPLE_FRACTION)
    
    # Train specific models
    if TRAIN_SPECIFIC:
        print("Training specific models...")
        trainer.train_all_models(symbols=SYMBOLS, use_transfer_learning=USE_TRANSFER_LEARNING)
    
    # Evaluate models
    if EVALUATE:
        print("Evaluating models...")
        results = trainer.evaluate_all_models(symbols=SYMBOLS)
        print("Evaluation results:")
        print(results)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 