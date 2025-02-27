import pandas as pd
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import joblib
from typing import Dict, List, Tuple, Optional, Union
import glob
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class SP500ModelTrainer:
    def __init__(self, processed_dir="data/processed", models_dir="models", 
                 sequence_length=60, batch_size=32, epochs=50):
        """
        Initialize the model trainer
        
        Args:
            processed_dir: Directory containing processed data
            models_dir: Directory to save trained models
            sequence_length: Number of time steps to use for sequence data
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = self._setup_logger()
        
        # Create necessary directories
        os.makedirs(os.path.join(self.models_dir, "general"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "specific"), exist_ok=True)
        
        # Features to use for modeling
        self.features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                         'Returns', 'MA_5', 'MA_20', 'Volatility']
        
        # Target variable
        self.target = 'Close'
        
        # Store encoders
        self.symbol_encoder = LabelEncoder()
        self.sector_encoder = LabelEncoder()
        
    def _setup_logger(self):
        logger = logging.getLogger('SP500ModelTrainer')
        logger.setLevel(logging.INFO)
        
        # Add file handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def create_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model
        
        Args:
            data: DataFrame containing the data
            target_col: Column name for the target variable
            
        Returns:
            X and y arrays for the model
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data.iloc[i:(i + self.sequence_length)][self.features].values)
            y.append(data.iloc[i + self.sequence_length][target_col])
            
        return np.array(X), np.array(y)
    
    def create_sequences_with_metadata(self, data: pd.DataFrame, target_col: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Create sequences with metadata for the general model
        
        Args:
            data: DataFrame containing the data
            target_col: Column name for the target variable
            
        Returns:
            Dictionary of input arrays and target array
        """
        X_seq, y = [], []
        X_symbol, X_sector = [], []
        
        # Group by symbol to ensure sequences don't cross between stocks
        self.logger.info(f"Creating sequences for {len(data['Symbol'].unique())} symbols")
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 100000
        total_rows = 0
        
        for symbol, group in data.groupby('Symbol'):
            self.logger.info(f"Processing sequences for {symbol}")
            group = group.sort_values('Date')
            
            # Process this symbol in chunks
            for i in range(0, len(group) - self.sequence_length, chunk_size):
                end_idx = min(i + chunk_size, len(group) - self.sequence_length)
                self.logger.info(f"Processing rows {i} to {end_idx} for {symbol}")
                
                for j in range(i, end_idx):
                    X_seq.append(group.iloc[j:(j + self.sequence_length)][self.features].values)
                    y.append(group.iloc[j + self.sequence_length][target_col])
                    
                    # Add metadata
                    symbol_id = self.symbol_encoder.transform([group.iloc[j]['Symbol']])[0]
                    sector_id = self.sector_encoder.transform([group.iloc[j]['Sector']])[0]
                    
                    X_symbol.append(symbol_id)
                    X_sector.append(sector_id)
                
                total_rows += (end_idx - i)
                self.logger.info(f"Created {total_rows} sequences so far")
        
        self.logger.info(f"Finished creating {len(y)} sequences")
        
        return {
            'sequence_input': np.array(X_seq),
            'symbol_input': np.array(X_symbol),
            'sector_input': np.array(X_sector)
        }, np.array(y)
    
    def train_general_model(self, sample_fraction=1.0) -> Model:
        """
        Train the general model on all stocks
        
        Args:
            sample_fraction: Fraction of data to use (0.1 = 10%)
            
        Returns:
            Trained Keras model
        """
        self.logger.info(f"=== TRAINING GENERAL MODEL (using {sample_fraction*100}% of data) ===")
        
        # Load the unified dataset
        unified_file = os.path.join(self.processed_dir, "general", "all_stocks_processed.csv")
        if not os.path.exists(unified_file):
            self.logger.error(f"Unified dataset not found at {unified_file}")
            return None
        
        self.logger.info(f"Loading unified dataset from {unified_file}")
        data = pd.read_csv(unified_file)
        self.logger.info(f"Loaded dataset with {len(data)} rows")
        
        # Sample the data if needed
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=42)
            self.logger.info(f"Sampled data: {len(data)} rows")
        
        # Fit encoders
        unique_symbols = data['Symbol'].unique()
        unique_sectors = data['Sector'].unique()
        self.logger.info(f"Found {len(unique_symbols)} unique symbols and {len(unique_sectors)} unique sectors")
        
        self.symbol_encoder.fit(unique_symbols)
        self.sector_encoder.fit(unique_sectors)
        
        # Save encoders
        encoder_path = os.path.join(self.models_dir, "general", "symbol_encoder.gz")
        self.logger.info(f"Saving symbol encoder to {encoder_path}")
        joblib.dump(self.symbol_encoder, encoder_path)
        
        encoder_path = os.path.join(self.models_dir, "general", "sector_encoder.gz")
        self.logger.info(f"Saving sector encoder to {encoder_path}")
        joblib.dump(self.sector_encoder, encoder_path)
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        self.logger.info(f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Create sequences
        self.logger.info("Creating sequence data for training...")
        X_train, y_train = self.create_sequences_with_metadata(train_data, self.target)
        self.logger.info(f"Created {len(y_train)} training sequences")
        
        # Create validation sequences
        self.logger.info("Creating sequence data for validation...")
        X_val, y_val = self.create_sequences_with_metadata(val_data, self.target)
        self.logger.info(f"Created {len(y_val)} validation sequences")
        
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
        plt.title('General Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.models_dir, "general", "training_history.png"))
        
        self.logger.info("General model training completed")
        
        return model
    
    def build_specific_model(self, input_shape: Tuple) -> Sequential:
        """
        Build a stock-specific model
        
        Args:
            input_shape: Shape of the input sequences
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train_general_model(self) -> Model:
        """
        Train the general model on all stocks
        
        Returns:
            Trained Keras model
        """
        self.logger.info("=== TRAINING GENERAL MODEL ===")
        
        # Load the unified dataset
        unified_file = os.path.join(self.processed_dir, "general", "all_stocks_processed.csv")
        if not os.path.exists(unified_file):
            self.logger.error(f"Unified dataset not found at {unified_file}")
            return None
        
        self.logger.info(f"Loading unified dataset from {unified_file}")
        data = pd.read_csv(unified_file)
        self.logger.info(f"Loaded dataset with {len(data)} rows")
        
        # Fit encoders
        unique_symbols = data['Symbol'].unique()
        unique_sectors = data['Sector'].unique()
        self.logger.info(f"Found {len(unique_symbols)} unique symbols and {len(unique_sectors)} unique sectors")
        
        self.symbol_encoder.fit(unique_symbols)
        self.sector_encoder.fit(unique_sectors)
        
        # Save encoders
        encoder_path = os.path.join(self.models_dir, "general", "symbol_encoder.gz")
        self.logger.info(f"Saving symbol encoder to {encoder_path}")
        joblib.dump(self.symbol_encoder, encoder_path)
        
        encoder_path = os.path.join(self.models_dir, "general", "sector_encoder.gz")
        self.logger.info(f"Saving sector encoder to {encoder_path}")
        joblib.dump(self.sector_encoder, encoder_path)
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        self.logger.info(f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Create sequences
        self.logger.info("Creating sequence data for training...")
        X_train, y_train = self.create_sequences_with_metadata(train_data, self.target)
        self.logger.info(f"Created {len(y_train)} training sequences")
        
        self.logger.info("Creating sequence data for validation...")
        X_val, y_val = self.create_sequences_with_metadata(val_data, self.target)
        self.logger.info(f"Created {len(y_val)} validation sequences")
        
        # Build model
        input_shape = (self.sequence_length, len(self.features))
        num_symbols = len(self.symbol_encoder.classes_)
        num_sectors = len(self.sector_encoder.classes_)
        
        self.logger.info(f"Building general model with input shape {input_shape}")
        self.logger.info(f"Using {num_symbols} symbols and {num_sectors} sectors for embeddings")
        
        model = self.build_general_model(input_shape, num_symbols, num_sectors)
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        # Callbacks
        model_path = os.path.join(self.models_dir, "general", "general_model.keras")
        self.logger.info(f"Model will be saved to {model_path}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.models_dir, "general", "training_log.csv"),
                separator=',', 
                append=False
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.models_dir, "general", "logs"),
                histogram_freq=1
            )
        ]
        
        # Train model
        self.logger.info(f"Starting training for up to {self.epochs} epochs with batch size {self.batch_size}")
        start_time = datetime.now()
        
        class LoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = datetime.now()
                self.logger.info(f"Epoch {epoch+1}/{self.params['epochs']} starting")
            
            def on_epoch_end(self, epoch, logs=None):
                duration = datetime.now() - self.epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{self.params['epochs']} completed in {duration}: "
                    f"loss={logs['loss']:.6f}, val_loss={logs['val_loss']:.6f}"
                )
        
        callbacks.append(LoggingCallback(self.logger))
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_duration = datetime.now() - start_time
        self.logger.info(f"Training completed in {training_duration}")
        
        # Plot training history
        self.logger.info("Generating training history plot")
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('General Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plot_path = os.path.join(self.models_dir, "general", "training_history.png")
        plt.savefig(plot_path)
        self.logger.info(f"Training history plot saved to {plot_path}")
        
        # Save final model
        final_model_path = os.path.join(self.models_dir, "general", "general_model_final.keras")
        model.save(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        self.logger.info("=== GENERAL MODEL TRAINING COMPLETED ===")
        
        return model
    
    def train_general_model(self, sample_fraction=0.1) -> Model:
        """
        Train the general model on all stocks
        
        Args:
            sample_fraction: Fraction of data to use (0.1 = 10%)
            
        Returns:
            Trained Keras model
        """
        self.logger.info(f"=== TRAINING GENERAL MODEL (using {sample_fraction*100}% of data) ===")
        
        # Load the unified dataset
        unified_file = os.path.join(self.processed_dir, "general", "all_stocks_processed.csv")
        if not os.path.exists(unified_file):
            self.logger.error(f"Unified dataset not found at {unified_file}")
            return None
        
        self.logger.info(f"Loading unified dataset from {unified_file}")
        data = pd.read_csv(unified_file)
        self.logger.info(f"Loaded dataset with {len(data)} rows")
        
        # Sample the data if needed
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=42)
            self.logger.info(f"Sampled data: {len(data)} rows")
        
        # Fit encoders
        unique_symbols = data['Symbol'].unique()
        unique_sectors = data['Sector'].unique()
        self.logger.info(f"Found {len(unique_symbols)} unique symbols and {len(unique_sectors)} unique sectors")
        
        self.symbol_encoder.fit(unique_symbols)
        self.sector_encoder.fit(unique_sectors)
        
        # Save encoders
        encoder_path = os.path.join(self.models_dir, "general", "symbol_encoder.gz")
        self.logger.info(f"Saving symbol encoder to {encoder_path}")
        joblib.dump(self.symbol_encoder, encoder_path)
        
        encoder_path = os.path.join(self.models_dir, "general", "sector_encoder.gz")
        self.logger.info(f"Saving sector encoder to {encoder_path}")
        joblib.dump(self.sector_encoder, encoder_path)
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        self.logger.info(f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Create sequences for training
        self.logger.info("Creating sequence data for training...")
        X_train, y_train = self.create_sequences_with_metadata(train_data, self.target)
        self.logger.info(f"Created {len(y_train)} training sequences")
        
        # Create sequences for validation
        self.logger.info("Creating sequence data for validation...")
        X_val, y_val = self.create_sequences_with_metadata(val_data, self.target)
        self.logger.info(f"Created {len(y_val)} validation sequences")
        
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
        plt.title('General Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.models_dir, "general", "training_history.png"))
        
        self.logger.info("General model training completed")
        
        return model
    
    def train_specific_model(self, symbol: str, use_transfer_learning: bool = True) -> Optional[Model]:
        """
        Train a stock-specific model
        
        Args:
            symbol: Stock symbol
            use_transfer_learning: Whether to use transfer learning from the general model
            
        Returns:
            Trained Keras model
        """
        self.logger.info(f"=== TRAINING SPECIFIC MODEL FOR {symbol} ===")
        
        # Find the stock's data file
        stock_files = []
        for sector_dir in os.listdir(os.path.join(self.processed_dir, "specific")):
            sector_path = os.path.join(self.processed_dir, "specific", sector_dir)
            if os.path.isdir(sector_path):
                stock_file = os.path.join(sector_path, f"{symbol}_processed.csv")
                if os.path.exists(stock_file):
                    stock_files.append(stock_file)
        
        if not stock_files:
            self.logger.error(f"No data file found for {symbol}")
            return None
        
        # Load data
        self.logger.info(f"Loading data from {stock_files[0]}")
        data = pd.read_csv(stock_files[0])
        self.logger.info(f"Loaded dataset with {len(data)} rows")
        
        # Split data
        train_data = data[data['Split'] == 'train']
        val_data = data[data['Split'] == 'val']
        self.logger.info(f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Create sequences
        self.logger.info("Creating sequence data for training...")
        X_train, y_train = self.create_sequences(train_data, self.target)
        self.logger.info(f"Created {len(y_train)} training sequences")
        
        self.logger.info("Creating sequence data for validation...")
        X_val, y_val = self.create_sequences(val_data, self.target)
        self.logger.info(f"Created {len(y_val)} validation sequences")
        
        # Build or load model
        input_shape = (self.sequence_length, len(self.features))
        self.logger.info(f"Input shape: {input_shape}")
        
        if use_transfer_learning:
            # Load general model
            general_model_path = os.path.join(self.models_dir, "general", "general_model.keras")
            if not os.path.exists(general_model_path):
                self.logger.warning("General model not found. Training specific model from scratch.")
                model = self.build_specific_model(input_shape)
            else:
                self.logger.info(f"Loading general model from {general_model_path} for transfer learning")
                custom_objects = {
                    'mse': tf.keras.losses.mean_squared_error
                }
                # Load general model
                general_model = load_model(general_model_path, custom_objects=custom_objects)
                #general_model = load_model(general_model_path)
                
                # Create a new model using the LSTM layers from the general model
                self.logger.info("Creating transfer learning model")
                
                # Create new model with transfer learning
                try:
                    # Create a simpler transfer learning approach
                    model = self.build_specific_model(input_shape)
                    
                    # Find LSTM layers in the general model
                    for i, layer in enumerate(general_model.layers):
                        if isinstance(layer, LSTM):
                            self.logger.info(f"Found LSTM layer: {layer.name}")
                            # Copy weights from general model to specific model
                            if i < len(model.layers):
                                if isinstance(model.layers[i], LSTM):
                                    self.logger.info(f"Copying weights to layer {i}")
                                    model.layers[i].set_weights(layer.get_weights())
                    
                    self.logger.info("Successfully created transfer learning model")
                except Exception as e:
                    self.logger.error(f"Error creating transfer learning model: {str(e)}")
                    self.logger.info("Falling back to training from scratch")
                    model = self.build_specific_model(input_shape)
        else:
            self.logger.info("Building specific model from scratch (no transfer learning)")
            model = self.build_specific_model(input_shape)
        
        # Add LoggingCallback
        class LoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = datetime.now()
                self.logger.info(f"Epoch {epoch+1}/{self.params['epochs']} starting")
            
            def on_epoch_end(self, epoch, logs=None):
                duration = datetime.now() - self.epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{self.params['epochs']} completed in {duration}: "
                    f"loss={logs['loss']:.6f}, val_loss={logs['val_loss']:.6f}"
                )
        
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        # Callbacks
        model_dir = os.path.join(self.models_dir, "specific", symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{symbol}_model.keras")
        self.logger.info(f"Model will be saved to {model_path}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, f"{symbol}_training_log.csv"),
                separator=',', 
                append=False
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, "logs"),
                histogram_freq=1
            ),
            LoggingCallback(self.logger)
        ]
        
        # Train model
        self.logger.info(f"Starting training for up to {self.epochs} epochs with batch size {self.batch_size}")
        start_time = datetime.now()
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_duration = datetime.now() - start_time
        self.logger.info(f"Training completed in {training_duration}")
        
        # Plot training history
        self.logger.info("Generating training history plot")
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{symbol} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plot_path = os.path.join(model_dir, f"{symbol}_training_history.png")
        plt.savefig(plot_path)
        self.logger.info(f"Training history plot saved to {plot_path}")
        
        # Save final model
        final_model_path = os.path.join(model_dir, f"{symbol}_model_final.keras")
        model.save(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        self.logger.info(f"=== SPECIFIC MODEL TRAINING FOR {symbol} COMPLETED ===")
        
        return model
    
    def train_all_models(self, symbols: Optional[List[str]] = None, use_transfer_learning: bool = True):
        """
        Train all models (general and specific)
        
        Args:
            symbols: List of symbols to train specific models for (None for all)
            use_transfer_learning: Whether to use transfer learning for specific models
        """
        # Train general model first
        self.logger.info("Starting the training process for all models")
        start_time = datetime.now()
        
        general_model = self.train_general_model()
        
        # If no symbols provided, find all available symbols
        if symbols is None:
            symbols = set()
            for sector_dir in os.listdir(os.path.join(self.processed_dir, "specific")):
                sector_path = os.path.join(self.processed_dir, "specific", sector_dir)
                if os.path.isdir(sector_path):
                    for file in os.listdir(sector_path):
                        if file.endswith("_processed.csv"):
                            symbol = file.split("_")[0]
                            symbols.add(symbol)
            
            symbols = list(symbols)
            self.logger.info(f"Found {len(symbols)} symbols for specific model training")
        
        # Train specific models
        total_symbols = len(symbols)
        for i, symbol in enumerate(symbols):
            self.logger.info(f"Training model {i+1}/{total_symbols} for {symbol}")
            try:
                self.train_specific_model(symbol, use_transfer_learning)
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {str(e)}")
        
        total_duration = datetime.now() - start_time
        self.logger.info(f"All model training completed in {total_duration}")

    def build_general_model(self, input_shape: Tuple, num_symbols: int, num_sectors: int) -> Model:
        """
        Build the general model for all stocks
        
        Args:
            input_shape: Shape of the input sequences
            num_symbols: Number of unique stock symbols
            num_sectors: Number of unique sectors
            
        Returns:
            Compiled Keras model
        """
        # Sequence input
        sequence_input = Input(shape=input_shape, name='sequence_input')
        
        # Symbol input
        symbol_input = Input(shape=(1,), name='symbol_input')
        symbol_embedding = Embedding(num_symbols, 10)(symbol_input)
        symbol_embedding = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), output_shape=(10,))(symbol_embedding)
        
        # Sector input
        sector_input = Input(shape=(1,), name='sector_input')
        sector_embedding = Embedding(num_sectors, 5)(sector_input)
        sector_embedding = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), output_shape=(5,))(sector_embedding)
        
        # LSTM layers for sequence processing
        lstm1 = LSTM(100, return_sequences=True)(sequence_input)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(50)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        
        # Combine sequence features with metadata
        combined = Concatenate()([dropout2, symbol_embedding, sector_embedding])
        
        # Dense layers
        dense1 = Dense(50, activation='relu')(combined)
        dense2 = Dense(25, activation='relu')(dense1)
        output = Dense(1)(dense2)
        
        # Create model
        model = Model(
            inputs=[sequence_input, symbol_input, sector_input],
            outputs=output
        )
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def evaluate_model(self, symbol: str, use_general: bool = False) -> Dict[str, float]:
        """
        Evaluate a model on test data
        
        Args:
            symbol: Stock symbol to evaluate
            use_general: Whether to use the general model instead of a specific one
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating model for {symbol}...")
        
        # Find the stock's data file
        stock_files = []
        for sector_dir in os.listdir(os.path.join(self.processed_dir, "specific")):
            sector_path = os.path.join(self.processed_dir, "specific", sector_dir)
            if os.path.isdir(sector_path):
                stock_file = os.path.join(sector_path, f"{symbol}_processed.csv")
                if os.path.exists(stock_file):
                    stock_files.append(stock_file)
        
        if not stock_files:
            self.logger.error(f"No data file found for {symbol}")
            return {}
        
        # Load data
        data = pd.read_csv(stock_files[0])
        
        # Get test data
        test_data = data[data['Split'] == 'test']
        if len(test_data) == 0:
            self.logger.warning(f"No test data found for {symbol}")
            return {}
            
        self.logger.info(f"Evaluating on {len(test_data)} test samples")
        
        # Create sequences
        X_test, y_test = self.create_sequences(test_data, self.target)
        
        # Load model
        if use_general:
            # Load general model and encoders
            model_path = os.path.join(self.models_dir, "general", "general_model.keras")
            if not os.path.exists(model_path):
                self.logger.error("General model not found")
                return {}
                
            self.logger.info(f"Loading general model from {model_path}")
            model = load_model(model_path)
            
            # Load encoders if not already loaded
            if not hasattr(self.symbol_encoder, 'classes_'):
                encoder_path = os.path.join(self.models_dir, "general", "symbol_encoder.gz")
                self.logger.info(f"Loading symbol encoder from {encoder_path}")
                self.symbol_encoder = joblib.load(encoder_path)
                
            if not hasattr(self.sector_encoder, 'classes_'):
                encoder_path = os.path.join(self.models_dir, "general", "sector_encoder.gz")
                self.logger.info(f"Loading sector encoder from {encoder_path}")
                self.sector_encoder = joblib.load(encoder_path)
            
            # Prepare inputs for general model
            sector = test_data['Sector'].iloc[0]
            
            # Transform inputs
            symbol_id = self.symbol_encoder.transform([symbol])[0]
            sector_id = self.sector_encoder.transform([sector])[0]
            
            # Create input dictionary
            X_test_dict = {
                'sequence_input': X_test,
                'symbol_input': np.full(len(X_test), symbol_id),
                'sector_input': np.full(len(X_test), sector_id)
            }
            
            # Make predictions
            self.logger.info("Making predictions with general model")
            y_pred = model.predict(X_test_dict)
        else:
            # Load specific model
            model_path = os.path.join(self.models_dir, "specific", symbol, f"{symbol}_model.keras")
            if not os.path.exists(model_path):
                self.logger.error(f"Specific model for {symbol} not found")
                return {}
                
            self.logger.info(f"Loading specific model from {model_path}")
            model = load_model(model_path)
            
            # Make predictions
            self.logger.info("Making predictions with specific model")
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_pred.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(y_pred.flatten() - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        y_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred.flatten())
        directional_accuracy = np.mean((y_diff > 0) == (y_pred_diff > 0))
        
        self.logger.info(f"Evaluation results for {symbol}:")
        self.logger.info(f"MSE: {mse:.6f}")
        self.logger.info(f"MAE: {mae:.6f}")
        self.logger.info(f"RMSE: {rmse:.6f}")
        self.logger.info(f"Directional Accuracy: {directional_accuracy:.6f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def evaluate_all_models(self, symbols: Optional[List[str]] = None, compare_with_general: bool = True) -> pd.DataFrame:
        """
        Evaluate all models
        
        Args:
            symbols: List of symbols to evaluate (None for all)
            compare_with_general: Whether to compare with the general model
            
        Returns:
            DataFrame with evaluation results
        """
        self.logger.info("=== EVALUATING ALL MODELS ===")
        start_time = datetime.now()
        
        # If no symbols provided, find all available symbols with trained models
        if symbols is None:
            symbols = []
            specific_dir = os.path.join(self.models_dir, "specific")
            if os.path.exists(specific_dir):
                for symbol_dir in os.listdir(specific_dir):
                    model_path = os.path.join(specific_dir, symbol_dir, f"{symbol_dir}_model.keras")
                    if os.path.exists(model_path):
                        symbols.append(symbol_dir)
            
            self.logger.info(f"Found {len(symbols)} trained models to evaluate")
        
        # Evaluate models
        results = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"Evaluating model {i+1}/{total_symbols} for {symbol}")
            
            try:
                # Evaluate specific model
                specific_metrics = self.evaluate_model(symbol, use_general=False)
                
                if compare_with_general:
                    # Evaluate general model on the same stock
                    general_metrics = self.evaluate_model(symbol, use_general=True)
                    
                    result = {
                        'Symbol': symbol,
                        'Specific_MSE': specific_metrics.get('mse', float('nan')),
                        'General_MSE': general_metrics.get('mse', float('nan')),
                        'Specific_MAE': specific_metrics.get('mae', float('nan')),
                        'General_MAE': general_metrics.get('mae', float('nan')),
                        'Specific_RMSE': specific_metrics.get('rmse', float('nan')),
                        'General_RMSE': general_metrics.get('rmse', float('nan')),
                        'Specific_DirectionalAccuracy': specific_metrics.get('directional_accuracy', float('nan')),
                        'General_DirectionalAccuracy': general_metrics.get('directional_accuracy', float('nan'))
                    }
                else:
                    result = {
                        'Symbol': symbol,
                        'MSE': specific_metrics.get('mse', float('nan')),
                        'MAE': specific_metrics.get('mae', float('nan')),
                        'RMSE': specific_metrics.get('rmse', float('nan')),
                        'DirectionalAccuracy': specific_metrics.get('directional_accuracy', float('nan'))
                    }
                    
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating model for {symbol}: {str(e)}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(self.models_dir, "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Evaluation results saved to {results_path}")
        
        # Calculate summary statistics
        if len(results) > 0:
            self.logger.info("=== EVALUATION SUMMARY ===")
            if compare_with_general:
                # Calculate average improvement
                results_df['MSE_Improvement'] = results_df['General_MSE'] - results_df['Specific_MSE']
                results_df['MAE_Improvement'] = results_df['General_MAE'] - results_df['Specific_MAE']
                results_df['RMSE_Improvement'] = results_df['General_RMSE'] - results_df['Specific_RMSE']
                results_df['DirectionalAccuracy_Improvement'] = results_df['Specific_DirectionalAccuracy'] - results_df['General_DirectionalAccuracy']
                
                # Log summary
                self.logger.info(f"Average MSE Improvement: {results_df['MSE_Improvement'].mean():.6f}")
                self.logger.info(f"Average MAE Improvement: {results_df['MAE_Improvement'].mean():.6f}")
                self.logger.info(f"Average RMSE Improvement: {results_df['RMSE_Improvement'].mean():.6f}")
                self.logger.info(f"Average Directional Accuracy Improvement: {results_df['DirectionalAccuracy_Improvement'].mean():.6f}")
                
                # Count wins
                mse_wins = (results_df['Specific_MSE'] < results_df['General_MSE']).sum()
                self.logger.info(f"Specific model better in MSE: {mse_wins}/{len(results_df)} ({mse_wins/len(results_df)*100:.2f}%)")
                
                dir_acc_wins = (results_df['Specific_DirectionalAccuracy'] > results_df['General_DirectionalAccuracy']).sum()
                self.logger.info(f"Specific model better in Directional Accuracy: {dir_acc_wins}/{len(results_df)} ({dir_acc_wins/len(results_df)*100:.2f}%)")
            else:
                self.logger.info(f"Average MSE: {results_df['MSE'].mean():.6f}")
                self.logger.info(f"Average MAE: {results_df['MAE'].mean():.6f}")
                self.logger.info(f"Average RMSE: {results_df['RMSE'].mean():.6f}")
                self.logger.info(f"Average Directional Accuracy: {results_df['DirectionalAccuracy'].mean():.6f}")
        
        evaluation_duration = datetime.now() - start_time
        self.logger.info(f"Evaluation completed in {evaluation_duration}")
        
        return results_df


def main():
    """
    Main function to train and evaluate models
    """
    # Create trainer
    trainer = SP500ModelTrainer(
        processed_dir="data/processed",
        models_dir="models",
        sequence_length=60,
        batch_size=32,
        epochs=2
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate S&P 500 stock prediction models')
    parser.add_argument('--train-general', action='store_true', help='Train the general model')
    parser.add_argument('--train-specific', action='store_true', help='Train specific models')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train/evaluate')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--no-transfer', action='store_true', help='Do not use transfer learning')
    
    args = parser.parse_args()
    
    # Default behavior if no arguments provided
    if not any(vars(args).values()):
        args.train_general = True
        args.train_specific = True
        args.evaluate = True
    
    # Train general model
    if args.train_general:
        trainer.train_general_model(sample_fraction=0.1)  # Use 10% of data

    # Train specific models
    if args.train_specific:
        trainer.train_all_models(symbols=args.symbols, use_transfer_learning=not args.no_transfer)
    
    # Evaluate models
    if args.evaluate:
        trainer.evaluate_all_models(symbols=args.symbols)


if __name__ == "__main__":
    main()