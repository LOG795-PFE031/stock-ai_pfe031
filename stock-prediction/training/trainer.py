"""
Model training module for stock prediction.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from ..models.factory import ModelFactory
from ..data.preprocessor import DataPreprocessor

class ModelTrainer:
    """Trainer for stock prediction models"""
    
    def __init__(self, config: Any):
        """
        Initialize the trainer
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        self.logger = logging.getLogger('ModelTrainer')
        self.model_factory = ModelFactory(config)
        self.preprocessor = DataPreprocessor(config)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = os.path.join(self.config.logging.LOG_DIR, "training")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def load_data(self, symbol: str, sector: str) -> Optional[pd.DataFrame]:
        """
        Load and preprocess data for a specific stock
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            
        Returns:
            Preprocessed DataFrame or None if loading fails
        """
        self.logger.info(f"Loading data for {symbol}")
        
        # Try to load preprocessed data first
        df = self.preprocessor.load_preprocessed_data(symbol, sector)
        if df is not None:
            self.logger.info(f"Loaded preprocessed data for {symbol}")
            return df
        
        # If no preprocessed data exists, load raw data and preprocess it
        data_file = os.path.join(self.config.data.RAW_DATA_DIR, "specific", sector, f"{symbol}.csv")
        if not os.path.exists(data_file):
            self.logger.error(f"No data file found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Preprocess the data
            df = self.preprocessor.preprocess(df)
            
            # Save preprocessed data
            self.preprocessor.save_preprocessed_data(df, symbol, sector)
            
            self.logger.info(f"Successfully loaded and preprocessed data for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def train_model(self, symbol: str, sector: str, model_type: str) -> bool:
        """
        Train a model for a specific stock
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            model_type: Type of model to train
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        self.logger.info(f"Training {model_type} model for {symbol}")
        
        # Load data
        df = self.load_data(symbol, sector)
        if df is None:
            return False
        
        try:
            # Create model
            model = self.model_factory.create_model(model_type)
            if model is None:
                self.logger.error(f"Failed to create {model_type} model")
                return False
            
            # Train model
            success = model.train(df)
            if not success:
                self.logger.error(f"Failed to train {model_type} model for {symbol}")
                return False
            
            # Save model
            model_dir = os.path.join(self.config.models.MODEL_DIR, "specific", sector)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{symbol}_{model_type}.h5")
            model.save(model_path)
            
            self.logger.info(f"Successfully trained and saved {model_type} model for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model for {symbol}: {str(e)}")
            return False
    
    def train_all_models(self, symbols: List[str], sectors: List[str], model_types: List[str]) -> Dict[str, List[str]]:
        """
        Train models for multiple stocks
        
        Args:
            symbols: List of stock symbols
            sectors: List of stock sectors
            model_types: List of model types to train
            
        Returns:
            Dictionary mapping model types to lists of successfully trained symbols
        """
        self.logger.info("Starting training for all models")
        
        results = {model_type: [] for model_type in model_types}
        
        for symbol, sector in zip(symbols, sectors):
            self.logger.info(f"Processing {symbol} ({sector})")
            
            for model_type in model_types:
                if self.train_model(symbol, sector, model_type):
                    results[model_type].append(symbol)
        
        # Log results
        for model_type, trained_symbols in results.items():
            self.logger.info(f"Successfully trained {model_type} models for {len(trained_symbols)} symbols")
            if trained_symbols:
                self.logger.info(f"Trained symbols: {', '.join(trained_symbols)}")
        
        return results 