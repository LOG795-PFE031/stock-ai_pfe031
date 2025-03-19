"""
LSTM model implementation for stock prediction.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, Any, Optional, Union, Tuple
import logging

from .base import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for stock price prediction"""
    
    def __init__(self, config: Any):
        """
        Initialize the LSTM model
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.sequence_length = config.model.SEQUENCE_LENGTH
        self.features = config.model.FEATURES
        self.batch_size = config.model.BATCH_SIZE
        self.epochs = config.model.EPOCHS
        self.logger = logging.getLogger('LSTMModel')
        
        # Set up GPU if available
        self.has_gpu = self.setup_gpu()
        
        # Create GPU monitoring callback if GPU is available
        self.gpu_monitor = self.GPUMonitorCallback(log_interval=2) if self.has_gpu else None
    
    def setup_gpu(self) -> bool:
        """Set up GPU for training if available"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU(s) available: {len(gpus)}")
                return True
            else:
                self.logger.info("No GPU available, using CPU")
                return False
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {str(e)}")
            return False
    
    class GPUMonitorCallback(tf.keras.callbacks.Callback):
        """Callback to monitor GPU memory usage"""
        def __init__(self, log_interval=1):
            super().__init__()
            self.log_interval = log_interval
            self.logger = logging.getLogger('GPUMonitor')
        
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.log_interval == 0:
                try:
                    gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                    self.logger.info(f"GPU Memory Usage - Current: {gpu_memory['current'] / 1024**2:.2f}MB, Peak: {gpu_memory['peak'] / 1024**2:.2f}MB")
                except Exception as e:
                    self.logger.error(f"Error monitoring GPU memory: {str(e)}")
    
    def build(self) -> None:
        """Build the LSTM model architecture with advanced configuration"""
        # Input layer with explicit dtype
        inputs = Input(shape=(self.sequence_length, len(self.features)), name='sequence_input', dtype=tf.float32)
        
        # Add BatchNormalization at input
        x = BatchNormalization(dtype=tf.float32)(inputs)
        
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
        
        # Dense layers with regularization
        x = Dense(32, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4))(x)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(16, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activity_regularizer=tf.keras.regularizers.l1(1e-4))(x)
        x = BatchNormalization(dtype=tf.float32)(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with optimized settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mape'],
            jit_compile=True  # Enable XLA compilation for better performance
        )
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model with input validation
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y) arrays
        """
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
                target = data.iloc[i + self.sequence_length]['Close']
                
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
        
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        self.logger.info(f"Created {len(X_array)} sequences with shape {X_array.shape}")
        
        # Final shape validation
        if X_array.shape[1:] != (self.sequence_length, len(self.features)):
            raise ValueError(f"Invalid final shape: {X_array.shape}. Expected (n, {self.sequence_length}, {len(self.features)})")
            
        return X_array, y_array
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model with optimized settings
        
        Args:
            train_data: Training data
            val_data: Optional validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        
        # Create validation sequences if provided
        if val_data is not None:
            X_val, y_val = self.create_sequences(val_data)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(self.config.data.MODELS_DIR, "lstm_model.weights.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        # Add GPU monitor callback if available
        if self.gpu_monitor:
            callbacks.append(self.gpu_monitor)
        
        # Create optimized tf.data.Dataset with larger shuffle buffer
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=100000)\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)\
                .batch(self.batch_size)\
                .prefetch(tf.data.AUTOTUNE)
            validation_data = val_dataset
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_data,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'history': history.history,
            'model': self.model
        }
    
    def predict(self, 
                data: pd.DataFrame,
                **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        # Create sequences
        X, _ = self.create_sequences(data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path where to save the model
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path from where to load the model
        """
        self.model = tf.keras.models.load_model(path)
    
    def evaluate(self, 
                 test_data: pd.DataFrame,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Create sequences
        X_test, y_test = self.create_sequences(test_data)
        
        # Evaluate model
        metrics = self.model.evaluate(X_test, y_test, return_dict=True)
        
        return metrics 