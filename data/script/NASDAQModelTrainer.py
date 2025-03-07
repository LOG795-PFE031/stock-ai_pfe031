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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Embedding
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
    BATCH_SIZE = 128
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    SAMPLE_FRACTION = 1.0
    FORCE_CPU_LSTM = True
    DEFAULT_NASDAQ_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CSCO', 'INTC']

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
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # Additional optimizations
            tf.config.optimizer.set_jit(True)
            
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
        """Build the general model for all stocks"""
        sequence_input = Input(shape=input_shape, name='sequence_input')
        
        symbol_input = Input(shape=(1,), name='symbol_input')
        symbol_embedding = Embedding(num_symbols, 10)(symbol_input)
        symbol_embedding = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=1),
            output_shape=(10,)
        )(symbol_embedding)
        
        sector_input = Input(shape=(1,), name='sector_input')
        sector_embedding = Embedding(num_sectors, 5)(sector_input)
        sector_embedding = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=1),
            output_shape=(5,)
        )(sector_embedding)
        
        lstm1 = LSTM(100, return_sequences=True)(sequence_input)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(50)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        
        combined = Concatenate()([dropout2, symbol_embedding, sector_embedding])
        
        dense1 = Dense(50, activation='relu')(combined)
        dense2 = Dense(25, activation='relu')(dense1)
        output = Dense(1)(dense2)
        
        model = Model(
            inputs=[sequence_input, symbol_input, sector_input],
            outputs=output
        )
        
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def build_specific_model(self, input_shape: Tuple) -> Model:
        """Build a stock-specific model using functional API for better shape handling"""
        # Use Input layer to explicitly specify input shape
        inputs = Input(shape=input_shape, name='sequence_input')
        
        x = LSTM(100, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='specific_stock_model')
        model.compile(optimizer='adam', loss='mse')
        
        self.logger.info(f"Built specific model with input shape {input_shape}")
        model.summary(print_fn=self.logger.info)
        
        return model
    
    def train_general_model(self, sample_fraction=1.0) -> Model:
        """Train the general model on all stocks"""
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
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.models_dir, "general", "general_model.keras"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
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
        """Train a stock-specific model"""
        self.logger.info(f"=== TRAINING NASDAQ SPECIFIC MODEL FOR {symbol} ===")
        
        # Find the stock's data file
        stock_file = os.path.join(self.processed_dir, "specific", "Technology", f"{symbol}_processed.csv")
        if not os.path.exists(stock_file):
            self.logger.error(f"No data file found for {symbol}")
            return None
        
        # Load and prepare data
        data = pd.read_csv(stock_file)
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, self.target)
        X_val, y_val = self.create_sequences(val_data, self.target)
        
        # Build or load model
        input_shape = (self.sequence_length, len(self.features))
        
        if use_transfer_learning:
            general_model_path = os.path.join(self.models_dir, "general", "general_model.keras")
            if os.path.exists(general_model_path):
                try:
                    general_model = load_model(general_model_path, safe_mode=False)
                    model = self.build_specific_model(input_shape)
                    
                    # Copy weights from general model's LSTM layers
                    for i, layer in enumerate(general_model.layers):
                        if isinstance(layer, LSTM):
                            if i < len(model.layers):
                                if isinstance(model.layers[i], LSTM):
                                    model.layers[i].set_weights(layer.get_weights())
                except Exception as e:
                    self.logger.error(f"Error in transfer learning: {e}")
                    model = self.build_specific_model(input_shape)
            else:
                model = self.build_specific_model(input_shape)
        else:
            model = self.build_specific_model(input_shape)
        
        # Callbacks
        model_dir = os.path.join(self.models_dir, "specific", symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(model_dir, f"{symbol}_model.keras"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        if self.has_gpu:
            callbacks.append(self.gpu_monitor)
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'NASDAQ {symbol} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f"{symbol}_training_history.png"))
        
        return model
    
    def train_all_models(self, symbols: Optional[List[str]] = None, use_transfer_learning: bool = True):
        """Train all models (general and specific) with better error handling"""
        start_time = datetime.now()
        
        if symbols and len(symbols) == 1:
            # If only one symbol is specified, skip general model training
            self.logger.info(f"Training specific model for {symbols[0]}")
            try:
                self.train_specific_model(symbols[0], use_transfer_learning)
            except Exception as e:
                self.logger.error(f"Error training model for {symbols[0]}: {e}")
        else:
            # Train general model first if multiple symbols or none specified
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
    
    if args.train_specific and args.symbols:
        # If specific symbols are provided, only train those
        trainer.train_all_models(
            symbols=args.symbols,
            use_transfer_learning=not args.no_transfer
        )
    else:
        # Default behavior
        trainer.train_all_models(
            symbols=Config.DEFAULT_NASDAQ_SYMBOLS if Config.FAST_MODE else None,
            use_transfer_learning=not args.no_transfer
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logging.error(f"ERROR: An exception occurred during training: {str(e)}")
        traceback.print_exc() 