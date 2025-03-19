"""
NASDAQ-100 Stock Prediction Model Training

This module combines all necessary functions for training NASDAQ-100 stock prediction models,
including GPU optimization and Colab support.
"""

import sys
import os
import time
import subprocess
from datetime import datetime
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Embedding, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
from typing import Dict, List, Tuple, Optional, Union
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm  # Add this import for progress bars

# Enable unsafe deserialization for Lambda layers
try:
    import keras
    keras.config.enable_unsafe_deserialization()
    print("Enabled unsafe deserialization for Lambda layers")
except Exception as e:
    print(f"Could not enable unsafe deserialization: {e}")

# Configuration settings - move these into the class
class Config:
    """Global configuration settings"""
    FAST_MODE = True
    MAX_SYMBOLS = 10
    BATCH_SIZE = 1024  # Increased from 512 for better GPU memory utilization
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    SAMPLE_FRACTION = 1.0
    FORCE_CPU_LSTM = False
    DEFAULT_NASDAQ_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CSCO', 'INTC']
    # Added memory configuration
    GPU_MEMORY_LIMIT = 14336  # 14GB to leave some headroom
    PREFETCH_BUFFER_SIZE = 8  # Increased from 4 for better data pipeline throughput

class NASDAQModelTrainer:
    def __init__(self, processed_dir="data/processed", models_dir="models", 
                 sequence_length=Config.SEQUENCE_LENGTH, 
                 batch_size=Config.BATCH_SIZE, 
                 epochs=Config.EPOCHS):
        """Initialize the model trainer"""
        self.config = Config  # Store config reference
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = self._setup_logger()
        
        # Create necessary directories
        os.makedirs(os.path.join(self.models_dir, "general"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "specific"), exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Features to use for modeling
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'MACD_Signal'
        ]
        
        # Target variable
        self.target = 'Close'
        
        # Store encoders
        self.symbol_encoder = LabelEncoder()
        self.sector_encoder = LabelEncoder()
        
        # Set up GPU if available
        self.has_gpu = self.setup_gpu()
        
        # Create GPU monitoring callback if GPU is available
        self.gpu_monitor = self.GPUMonitorCallback(log_interval=2) if self.has_gpu else None
    
    def _setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger('NASDAQTrainer')
        logger.setLevel(logging.INFO)
        
        # Add file handler
        fh = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def setup_gpu(self) -> bool:
        """Configure GPU settings for optimal performance"""
        self.logger.info("Setting up GPU...")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            self.logger.warning("No GPU found. Training will run on CPU.")
            return False
        
        self.logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        try:
            # Enable memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set TensorFlow memory allocation configuration
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=self.config.GPU_MEMORY_LIMIT)]
            )
            
            # Enable mixed precision training for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Enable XLA optimization
            tf.config.optimizer.set_jit(True)
            
            # Optimize thread settings
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(8)
            
            # Print GPU info
            self.logger.info("\nGPU Information:")
            os.system('nvidia-smi')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {e}")
            return False
    
    class GPUMonitorCallback(tf.keras.callbacks.Callback):
        """Callback to monitor GPU usage during training"""
        def __init__(self, log_interval=5):
            super().__init__()
            self.log_interval = log_interval
            self.start_time = None
        
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print("Starting training with GPU monitoring...")
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.log_interval == 0:
                print("\nGPU Statistics:")
                os.system('nvidia-smi')
                elapsed = time.time() - self.start_time
                print(f"Time elapsed: {elapsed:.2f} seconds")
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                print(f"Training metrics: {metrics_str}")
    
    def create_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM model with input validation"""
        self.logger.info(f"Creating sequences from {len(data)} rows of data")
        
        if len(data) <= self.sequence_length:
            raise ValueError(f"Insufficient data: {len(data)} rows. Need at least {self.sequence_length + 1} rows.")
        
        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            try:
                # Extract sequence
                sequence = data.iloc[i:(i + self.sequence_length)][self.features].values
                target = data.iloc[i + self.sequence_length][target_col]
                
                # Validate sequence shape
                if sequence.shape != (self.sequence_length, len(self.features)):
                    self.logger.warning(f"Invalid sequence shape at index {i}: {sequence.shape}")
                    continue
                
                X.append(sequence)
                y.append(target)
                
            except Exception as e:
                self.logger.error(f"Error creating sequence at index {i}: {str(e)}")
                continue
        
        if not X:
            raise ValueError("No valid sequences could be created from the data")
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        self.logger.info(f"Created {len(X_array)} sequences with shape {X_array.shape}")
        
        # Final shape validation
        if X_array.shape[1:] != (self.sequence_length, len(self.features)):
            raise ValueError(f"Invalid final shape: {X_array.shape}. Expected (n, {self.sequence_length}, {len(self.features)})")
            
        return X_array, y_array
    
    def create_sequences_with_metadata(self, data: pd.DataFrame, target_col: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Create sequences with metadata for the general model with progress tracking and memory optimization"""
        self.logger.info(f"Creating sequences for {len(data['Symbol'].unique())} symbols")
        
        # Pre-allocate lists with approximate size to avoid resizing
        estimated_sequences = len(data) - self.sequence_length * len(data['Symbol'].unique())
        X_seq = []
        y = []
        X_symbol = []
        X_sector = []
        
        # Process in smaller chunks by symbol
        for symbol in tqdm(data['Symbol'].unique(), desc="Processing symbols"):
            try:
                # Get data for this symbol
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
                
                if len(symbol_data) <= self.sequence_length:
                    self.logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                # Process in chunks to manage memory
                chunk_size = 10000  # Adjust this value based on available memory
                
                for i in range(0, len(symbol_data) - self.sequence_length, chunk_size):
                    end_idx = min(i + chunk_size, len(symbol_data) - self.sequence_length)
                    
                    # Create sequences for this chunk
                    for j in range(i, end_idx):
                        sequence = symbol_data.iloc[j:(j + self.sequence_length)][self.features].values
                        target = symbol_data.iloc[j + self.sequence_length][target_col]
                        
                        X_seq.append(sequence)
                        y.append(target)
                        
                        # Add metadata
                        symbol_id = self.symbol_encoder.transform([symbol])[0]
                        sector_id = self.sector_encoder.transform([symbol_data.iloc[j]['Sector']])[0]
                        
                        X_symbol.append(symbol_id)
                        X_sector.append(sector_id)
                    
                    # Clear some memory
                    if len(X_seq) >= chunk_size:
                        self.logger.info(f"Created {len(X_seq)} sequences so far")
                        
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Total sequences created: {len(y)}")
        
        # Convert to numpy arrays in chunks to manage memory
        sequence_array = np.array(X_seq)
        symbol_array = np.array(X_symbol)
        sector_array = np.array(X_sector)
        target_array = np.array(y)
        
        return {
            'sequence_input': sequence_array,
            'symbol_input': symbol_array,
            'sector_input': sector_array
        }, target_array
    
    def build_general_model(self, input_shape: Tuple, num_symbols: int, num_sectors: int) -> Model:
        """Build the general model with optimized configuration and strong regularization"""
        # Input layers with explicit dtype
        sequence_input = Input(shape=input_shape, name='sequence_input', dtype=tf.float32)
        
        # Add BatchNormalization at input
        x = BatchNormalization(dtype=tf.float32)(sequence_input)
        
        # First LSTM layer with reduced units and strong regularization
        x = LSTM(64, return_sequences=True,  # Reduced from 100
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4),
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                dtype=tf.float32)(x)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.4)(x)  # Increased from 0.2
        
        # Second LSTM layer with reduced units and strong regularization
        x = LSTM(32,  # Reduced from 50
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4),
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                dtype=tf.float32)(x)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.4)(x)  # Increased from 0.2
        
        # Symbol embedding with regularization
        symbol_input = Input(shape=(1,), name='symbol_input', dtype=tf.int32)
        symbol_embedding = Embedding(num_symbols, 8,  # Reduced embedding dim from 10 to 8
                                  embeddings_regularizer=tf.keras.regularizers.l2(1e-3))(symbol_input)
        symbol_embedding = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=1),
            output_shape=(8,)  # Updated to match new embedding dim
        )(symbol_embedding)
        symbol_embedding = Dropout(0.3)(symbol_embedding)
        
        # Sector embedding with regularization
        sector_input = Input(shape=(1,), name='sector_input', dtype=tf.int32)
        sector_embedding = Embedding(num_sectors, 4,  # Reduced embedding dim from 5 to 4
                                  embeddings_regularizer=tf.keras.regularizers.l2(1e-3))(sector_input)
        sector_embedding = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=1),
            output_shape=(4,)  # Updated to match new embedding dim
        )(sector_embedding)
        sector_embedding = Dropout(0.3)(sector_embedding)
        
        # Combine features with BatchNormalization
        combined = Concatenate()([x, symbol_embedding, sector_embedding])
        combined = BatchNormalization(dtype=tf.float32)(combined)
        
        # Dense layers with reduced capacity and strong regularization
        x = Dense(32, activation='relu',  # Reduced from 50
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4),
                dtype=tf.float32)(combined)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(16, activation='relu',  # Reduced from 25
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4),
                dtype=tf.float32)(x)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1, dtype=tf.float32)(x)
        
        model = Model(
            inputs=[sequence_input, symbol_input, sector_input],
            outputs=outputs,
            name='general_stock_model'
        )
        
        # Use a lower initial learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,  # Reduced from 0.001
            clipnorm=1.0  # Add gradient clipping
        )
        
        # Use Huber loss instead of MSE for better robustness
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_specific_model(self, input_shape: Tuple) -> Model:
        """Build a stock-specific model using functional API with enhanced regularization"""
        # Ensure input data type is float32
        inputs = Input(shape=input_shape, name='sequence_input', dtype=tf.float32)
        
        # Add BatchNormalization at input with higher momentum
        x = BatchNormalization(momentum=0.99, dtype=tf.float32)(inputs)
        
        # First LSTM layer with reduced complexity
        x = LSTM(24, return_sequences=True,  # Reduced from 32
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                dtype=tf.float32)(x)
        x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = Dropout(0.2)(x)  # Reduced from 0.3
        
        # Second LSTM layer
        x = LSTM(12,  # Reduced from 16
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                dtype=tf.float32)(x)
        x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = Dropout(0.2)(x)  # Reduced from 0.3
        
        # Dense layers with ELU activation for better gradient flow
        x = Dense(12, activation='elu',  # Changed from selu to elu
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                dtype=tf.float32)(x)
        x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = Dropout(0.1)(x)  # Reduced from 0.2
        
        x = Dense(6, activation='elu',  # Changed from selu to elu
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                kernel_initializer='glorot_uniform',
                dtype=tf.float32)(x)
        x = BatchNormalization(momentum=0.99, dtype=tf.float32)(x)
        x = Dropout(0.1)(x)  # Reduced from 0.2
        
        outputs = Dense(1, activation='linear', dtype=tf.float32)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='specific_stock_model')
        
        # Use a higher initial learning rate with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Increased from 0.0005
            clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Use MSE loss for better stability in early training
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Changed from Huber loss
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
        
        self.logger.info(f"Built specific model with input shape {input_shape}")
        model.summary(print_fn=self.logger.info)
        
        return model
    
    def train_general_model(self, sample_fraction=1.0) -> Model:
        """Train the general model with optimized settings"""
        self.logger.info(f"=== TRAINING NASDAQ GENERAL MODEL (using {sample_fraction*100}% of data) ===")
        
        # Load the unified dataset
        unified_file = os.path.join(self.processed_dir, "general", "nasdaq_stocks_processed.csv")
        if not os.path.exists(unified_file):
            self.logger.error(f"Unified dataset not found at {unified_file}")
            return None
        
        data = pd.read_csv(unified_file)
        
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=42)
        
        # Fit encoders
        self.symbol_encoder.fit(data['Symbol'].unique())
        self.sector_encoder.fit(data['Sector'].unique())
        
        # Save encoders
        joblib.dump(self.symbol_encoder, os.path.join(self.models_dir, "general", "symbol_encoder.gz"))
        joblib.dump(self.sector_encoder, os.path.join(self.models_dir, "general", "sector_encoder.gz"))
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        
        # Create sequences
        X_train, y_train = self.create_sequences_with_metadata(train_data, self.target)
        X_val, y_val = self.create_sequences_with_metadata(val_data, self.target)
        
        # Build model
        input_shape = (self.sequence_length, len(self.features))
        num_symbols = len(self.symbol_encoder.classes_)
        num_sectors = len(self.sector_encoder.classes_)
        
        model = self.build_general_model(input_shape, num_symbols, num_sectors)
        
        # Add data prefetching for better GPU utilization
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
            
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
        
        # Add callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.models_dir, "general", "general_model.weights.h5"),  # Changed file extension
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Recompile model without jit_compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Train model with optimized settings
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('NASDAQ General Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.models_dir, "general", "training_history.png"))
        
        return model
    
    def train_specific_model(self, symbol: str, use_transfer_learning: bool = True) -> Optional[Model]:
        """Train a stock-specific model with enhanced training configuration"""
        self.logger.info(f"=== TRAINING NASDAQ SPECIFIC MODEL FOR {symbol} ===")
        
        # Find the stock's data file
        stock_file = os.path.join(self.processed_dir, "specific", "Technology", f"{symbol}_processed.csv")
        if not os.path.exists(stock_file):
            self.logger.error(f"No data file found for {symbol}")
            return None
        
        # Load and prepare data
        data = pd.read_csv(stock_file)
        
        # Split data with validation split
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, self.target)
        X_val, y_val = self.create_sequences(val_data, self.target)
        
        # Convert data to float32
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        
        # Build or load model
        input_shape = (self.sequence_length, len(self.features))
        model = self.build_specific_model(input_shape)
        
        # Create optimized tf.data.Dataset with larger shuffle buffer
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=100000)\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
        
        # Enhanced callbacks with modified configuration
        model_dir = os.path.join(self.models_dir, "specific", symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-4,
                mode='min'
            ),
            ModelCheckpoint(
                os.path.join(model_dir, f"{symbol}_model.weights.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                min_delta=1e-4,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr * 0.95 if epoch > 10 else lr,
                verbose=1
            )
        ]
        
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Train model with modified settings
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot enhanced training history
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'NASDAQ {symbol} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'NASDAQ {symbol} Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"{symbol}_training_history.png"))
        
        return model
    
    def train_all_models(self, symbols: Optional[List[str]] = None, use_transfer_learning: bool = True):
        """Train all models (general and specific) with better error handling"""
        start_time = datetime.now()
        
        if symbols and len(symbols) > 0:
            # If specific symbols are provided, only train those models
            self.logger.info(f"Training specific models for symbols: {symbols}")
            for symbol in symbols:
                try:
                    self.train_specific_model(symbol, use_transfer_learning)
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {e}")
        else:
            # Train general model first if no specific symbols provided
            try:
                self.train_general_model()
            except Exception as e:
                self.logger.error(f"Error training general model: {e}")
            
            # Get symbols if not provided
            if symbols is None:
                symbols = self.config.DEFAULT_NASDAQ_SYMBOLS if self.config.FAST_MODE else self.get_available_symbols()
            
            # Train specific models
            for i, symbol in enumerate(symbols):
                self.logger.info(f"Training model {i+1}/{len(symbols)} for {symbol}")
                try:
                    self.train_specific_model(symbol, use_transfer_learning)
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {e}")
        
        self.logger.info(f"All model training completed in {datetime.now() - start_time}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from processed data directory"""
        symbols = []
        tech_dir = os.path.join(self.processed_dir, "specific", "Technology")
        if os.path.exists(tech_dir):
            for file in os.listdir(tech_dir):
                if file.endswith("_processed.csv"):
                    symbols.append(file.split("_")[0])
        return symbols if not self.config.FAST_MODE else self.config.DEFAULT_NASDAQ_SYMBOLS[:self.config.MAX_SYMBOLS]

def main():
    """Main function with improved argument handling"""
    parser = argparse.ArgumentParser(description='Train NASDAQ stock prediction models')
    parser.add_argument('--train-general', action='store_true', help='Train the general model')
    parser.add_argument('--train-specific', action='store_true', help='Train specific models')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--fast', action='store_true', help='Use fast mode')
    parser.add_argument('--no-transfer', action='store_true', help='Disable transfer learning')
    
    args = parser.parse_args()
    
    # Update config based on arguments
    Config.FAST_MODE = args.fast
    
    trainer = NASDAQModelTrainer(
        processed_dir="data/processed",
        models_dir="models",
        sequence_length=Config.SEQUENCE_LENGTH,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS
    )
    
    if args.train_specific:
        # Only train specific models
        symbols = args.symbols if args.symbols else (
            Config.DEFAULT_NASDAQ_SYMBOLS if Config.FAST_MODE 
            else trainer.get_available_symbols()
        )
        trainer.train_all_models(
            symbols=symbols,
            use_transfer_learning=not args.no_transfer
        )
    elif args.train_general:
        # Only train general model
        trainer.train_general_model()
    else:
        # Default behavior - train both with all symbols
        trainer.train_all_models(
            symbols=None,
            use_transfer_learning=not args.no_transfer
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logging.error(f"ERROR: An exception occurred during training: {str(e)}")
        traceback.print_exc() 