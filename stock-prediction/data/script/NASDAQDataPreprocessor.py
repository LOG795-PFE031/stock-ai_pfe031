import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Dict, List, Tuple
import glob
import json

class NASDAQDataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed", models_dir="models/specific"):
        """
        Initialize the preprocessor

        Args:
            raw_dir: Directory containing raw downloaded data
            processed_dir: Directory to save processed data
            models_dir: Directory to save scalers and other preprocessing models
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.logger = self._setup_logger()

        # Create necessary directories
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, "general"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, "specific"), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Features to use for modeling
        self.features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Store scalers for each stock
        self.scalers = {}

    def _setup_logger(self):
        logger = logging.getLogger('NASDAQPreprocessor')
        logger.setLevel(logging.INFO)

        # Add file handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/preprocess_nasdaq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        return logger

    def preprocess_stock(self, file_path: str, is_unified: bool = False) -> Tuple[pd.DataFrame, str, str]:
        """
        Preprocess a single stock's data

        Args:
            file_path: Path to the stock's CSV file
            is_unified: Whether this is for the unified/general model

        Returns:
            Processed dataframe, symbol, and sector
        """
        try:
            # Load data
            df = pd.read_csv(file_path)

            # Skip the problematic second row if it exists
            if df.iloc[0].isna().any() or (df.iloc[0] == df.columns).any():
                df = df.iloc[1:].reset_index(drop=True)

            # Extract symbol from filename
            filename = os.path.basename(file_path)
            symbol = filename.split('_')[0]  # SYMBOL_stock_price.csv format

            # Get sector (should be Technology for all NASDAQ stocks in this case)
            sector = "Technology"

            self.logger.info(f"Processing {symbol} in sector {sector}")

            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Sort by date
            df = df.sort_values('Date')

            # Handle missing values
            df = df.dropna()

            # Ensure numeric data types for price columns
            for col in self.features:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop any rows with NaN after conversion
            df = df.dropna(subset=self.features)

            # Handle outliers using IQR method for each feature
            for col in self.features:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3 instead of 1.5 for less aggressive filtering
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            # Calculate returns with winsorization
            df['Returns'] = df['Close'].pct_change()
            df['Returns'] = df['Returns'].clip(
                lower=df['Returns'].quantile(0.01),
                upper=df['Returns'].quantile(0.99)
            )

            # Calculate technical indicators with smoothing
            df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std()

            # Add momentum indicators with smoothing
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])

            # Normalize returns and volatility separately
            df['Returns'] = (df['Returns'] - df['Returns'].mean()) / df['Returns'].std()
            df['Volatility'] = (df['Volatility'] - df['Volatility'].mean()) / df['Volatility'].std()

            # Drop rows with NaN from the new calculations
            df = df.dropna()

            # Create train/validation/test splits with proper chronological order
            train_end = int(len(df) * 0.8)
            val_end = int(len(df) * 0.9)

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

            # Scale the data
            scaler = MinMaxScaler()

            # Select features for scaling
            scale_features = self.features + ['Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'MACD', 'MACD_Signal']

            # Prepare data for scaling with proper feature names
            train_features = train_df[scale_features].copy()
            val_features = val_df[scale_features].copy()
            test_features = test_df[scale_features].copy()

            # Fit scaler on training data and transform
            train_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                columns=scale_features,
                index=train_features.index
            )
            
            # Transform validation and test data
            val_scaled = pd.DataFrame(
                scaler.transform(val_features),
                columns=scale_features,
                index=val_features.index
            )
            test_scaled = pd.DataFrame(
                scaler.transform(test_features),
                columns=scale_features,
                index=test_features.index
            )

            # Create directory for the symbol first
            symbol_dir = os.path.join(self.models_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)

            try:
                # Save scaling metadata for debugging
                scaling_metadata = {
                    'feature_order': scale_features,
                    'min_values': scaler.data_min_.tolist(),
                    'max_values': scaler.data_max_.tolist(),
                    'feature_names': scale_features,
                    'sample_original': train_features.iloc[0].to_dict(),  # Changed to dict for better readability
                    'sample_scaled': train_scaled.iloc[0].to_dict()  # Changed to dict for better readability
                }
                
                # Save the metadata alongside the scaler
                metadata_path = os.path.join(symbol_dir, f"{symbol}_scaler_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(scaling_metadata, f, indent=4)

                # Save the scaler
                scaler_path = os.path.join(symbol_dir, f"{symbol}_scaler.gz")
                joblib.dump(scaler, scaler_path)
                self.scalers[symbol] = scaler
                
                self.logger.info(f"Successfully saved scaler and metadata for {symbol}")
            except Exception as e:
                self.logger.error(f"Error saving scaler data for {symbol}: {str(e)}")
                raise

            # Add date and metadata back
            for split_df in [train_scaled, val_scaled, test_scaled]:
                split_df['Date'] = df.loc[split_df.index, 'Date']
                split_df['Symbol'] = symbol
                split_df['Sector'] = sector

            # Add split indicators
            train_scaled['Split'] = 'train'
            val_scaled['Split'] = 'val'
            test_scaled['Split'] = 'test'

            # Combine into one dataframe
            combined_df = pd.concat([train_scaled, val_scaled, test_scaled])

            return combined_df, symbol, sector

        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {str(e)}")
            return None, None, None

    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def preprocess_all_stocks(self):
        """Preprocess all NASDAQ stocks"""
        self.logger.info("Processing NASDAQ stock data...")

        # Get all stock files from the Technology sector
        stock_files = glob.glob(os.path.join(self.raw_dir, "Technology", "*_stock_price.csv"))

        self.logger.info(f"Found {len(stock_files)} stock files to process")

        all_stocks_data = []

        for file in stock_files:
            processed_df, symbol, sector = self.preprocess_stock(file)

            if processed_df is not None:
                # Save individual stock data
                output_dir = os.path.join(self.processed_dir, "specific", sector)
                os.makedirs(output_dir, exist_ok=True)

                if symbol is not None:
                    output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
                    processed_df.to_csv(output_file, index=False)

                    # Add to the list for the general model
                    all_stocks_data.append(processed_df)

                    self.logger.info(f"Successfully processed {symbol}")

        # Create unified dataset
        if all_stocks_data:
            self.logger.info("Creating unified dataset for general model...")
            unified_df = pd.concat(all_stocks_data, ignore_index=True)

            # Save the unified dataset
            unified_file = os.path.join(self.processed_dir, "general", "nasdaq_stocks_processed.csv")
            unified_df.to_csv(unified_file, index=False)

            self.logger.info(f"Unified dataset created with {len(unified_df)} rows")

        self.logger.info("Preprocessing completed")

def main():
    preprocessor = NASDAQDataPreprocessor(
        raw_dir="data/raw",
        processed_dir="data/processed",
        models_dir="models/specific"
    )
    preprocessor.preprocess_all_stocks()

if __name__ == "__main__":
    main()