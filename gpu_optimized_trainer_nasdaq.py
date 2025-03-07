"""
GPU-Optimized NASDAQ-100 Stock Prediction Model Trainer

This module extends the NASDAQModelTrainer class with GPU optimizations
for faster training on Google Colab or other GPU-enabled environments.
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from typing import List, Optional, Dict, Any, Tuple

# Import the original trainer class
from train_nasdaq_models import NASDAQModelTrainer

class GPUOptimizedTrainerNASDAQ(NASDAQModelTrainer):
    """
    GPU-optimized version of the NASDAQModelTrainer class
    with additional features for GPU utilization monitoring and performance.
    """
    
    def __init__(self, processed_dir="data/processed", models_dir="models/specific", 
                 sequence_length=60, batch_size=128, epochs=50):
        """Initialize with NASDAQ-specific paths and larger batch size for GPU"""
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
    
    # Rest of the code remains the same as gpu_optimized_trainer.py
    # Just update logging messages to reference NASDAQ instead of S&P 500 