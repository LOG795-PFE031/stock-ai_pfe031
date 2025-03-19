"""
Data preprocessing module for stock prediction models.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import joblib

class DataPreprocessor:
    """Preprocessor for stock prediction data"""
    
    def __init__(self, config: Any):
        """
        Initialize the preprocessor
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger('DataPreprocessor')
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        # Technical indicators parameters
        self.ma_periods = [5, 20, 50, 200]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.volatility_period = 20
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        self.logger.info("Calculating technical indicators")
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate moving averages
        for period in self.ma_periods:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=self.volatility_period).std()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Calculate volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Calculate price momentum
        df['Momentum'] = df['Close'].pct_change(periods=10)
        
        # Calculate ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        self.logger.info("Technical indicators calculated successfully")
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        self.logger.info("Handling outliers")
        
        # Calculate z-scores for price and volume
        price_zscore = np.abs((df['Close'] - df['Close'].mean()) / df['Close'].std())
        volume_zscore = np.abs((df['Volume'] - df['Volume'].mean()) / df['Volume'].std())
        
        # Identify outliers (z-score > 3)
        price_outliers = price_zscore > 3
        volume_outliers = volume_zscore > 3
        
        # Replace outliers with rolling median
        df.loc[price_outliers, 'Close'] = df['Close'].rolling(window=5, center=True).median()
        df.loc[volume_outliers, 'Volume'] = df['Volume'].rolling(window=5, center=True).median()
        
        # Handle outliers in technical indicators
        for col in df.columns:
            if col not in ['Date', 'Symbol', 'Sector']:
                zscore = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = zscore > 3
                df.loc[outliers, col] = df[col].rolling(window=5, center=True).median()
        
        self.logger.info("Outliers handled successfully")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        self.logger.info("Normalizing data")
        
        # Normalize price data
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            if col in df.columns:
                df[col] = self.price_scaler.fit_transform(df[[col]])
        
        # Normalize volume
        if 'Volume' in df.columns:
            df['Volume'] = self.volume_scaler.fit_transform(df[['Volume']])
        
        # Normalize technical indicators
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Sector'] + price_cols + ['Volume']]
        if feature_cols:
            df[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols])
        
        self.logger.info("Data normalized successfully")
        return df
    
    def create_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create train/validation/test splits
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with split information
        """
        self.logger.info("Creating data splits")
        
        # Calculate split indices
        train_end = int(len(df) * 0.8)
        val_end = int(len(df) * 0.9)
        
        # Create split column
        df['Split'] = 'train'
        df.loc[train_end:val_end, 'Split'] = 'val'
        df.loc[val_end:, 'Split'] = 'test'
        
        self.logger.info("Data splits created successfully")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting data preprocessing")
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Handle outliers
        df = self.handle_outliers(df)
        
        # Normalize data
        df = self.normalize_data(df)
        
        # Create splits
        df = self.create_splits(df)
        
        self.logger.info("Data preprocessing completed successfully")
        return df
    
    def save_preprocessed_data(self, df: pd.DataFrame, symbol: str, sector: str) -> None:
        """
        Save preprocessed data
        
        Args:
            df: Preprocessed DataFrame
            symbol: Stock symbol
            sector: Stock sector
        """
        # Create sector directory if it doesn't exist
        sector_dir = os.path.join(self.config.data.PROCESSED_DATA_DIR, "specific", sector)
        os.makedirs(sector_dir, exist_ok=True)
        
        # Save preprocessed data
        output_file = os.path.join(sector_dir, f"{symbol}_processed.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Preprocessed data saved to {output_file}")
        
        # Save scalers
        scalers_dir = os.path.join(self.config.data.PROCESSED_DATA_DIR, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        
        scalers = {
            'price_scaler': self.price_scaler,
            'volume_scaler': self.volume_scaler,
            'feature_scaler': self.feature_scaler
        }
        
        for name, scaler in scalers.items():
            scaler_file = os.path.join(scalers_dir, f"{symbol}_{name}.joblib")
            joblib.dump(scaler, scaler_file)
            self.logger.info(f"Scaler saved to {scaler_file}")
    
    def load_preprocessed_data(self, symbol: str, sector: str) -> Optional[pd.DataFrame]:
        """
        Load preprocessed data
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            
        Returns:
            Preprocessed DataFrame or None if not found
        """
        # Try to find the stock's data file
        data_file = os.path.join(self.config.data.PROCESSED_DATA_DIR, "specific", sector, f"{symbol}_processed.csv")
        
        if not os.path.exists(data_file):
            self.logger.error(f"No preprocessed data found for {symbol}")
            return None
        
        # Load preprocessed data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Load scalers
        scalers_dir = os.path.join(self.config.data.PROCESSED_DATA_DIR, "scalers")
        
        scalers = {
            'price_scaler': 'price_scaler',
            'volume_scaler': 'volume_scaler',
            'feature_scaler': 'feature_scaler'
        }
        
        for attr, name in scalers.items():
            scaler_file = os.path.join(scalers_dir, f"{symbol}_{name}.joblib")
            if os.path.exists(scaler_file):
                setattr(self, attr, joblib.load(scaler_file))
                self.logger.info(f"Loaded {name} from {scaler_file}")
        
        return df 