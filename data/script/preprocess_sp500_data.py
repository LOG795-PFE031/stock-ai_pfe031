import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Dict, List, Tuple
import glob

class SP500DataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed", models_dir="models"):
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
        logger = logging.getLogger('SP500Preprocessor')
        logger.setLevel(logging.INFO)
        
        # Add file handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
            
            # Extract symbol from filename if not in data
            filename = os.path.basename(file_path)
            file_symbol = filename.split('_')[0]  # Assuming format is SYMBOL_stock_price.csv
            
            # Extract symbol and sector from data or use filename
            if 'Symbol' in df.columns and not pd.isna(df['Symbol'].iloc[0]):
                symbol = df['Symbol'].iloc[0]
            else:
                symbol = file_symbol
                
            # Get sector from directory structure if not in data
            if 'Sector' in df.columns and not pd.isna(df['Sector'].iloc[0]):
                sector = df['Sector'].iloc[0]
            else:
                # Extract sector from file path
                # Assuming path structure: data/raw/SECTOR/SYMBOL_stock_price.csv
                path_parts = file_path.split(os.sep)
                if len(path_parts) >= 3:
                    sector = path_parts[-2]  # Second-to-last element should be the sector
                else:
                    sector = "Unknown"
            
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
            
            # Calculate additional features if needed
            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Drop rows with NaN from the new calculations
            df = df.dropna()
            
            # For the unified model, add normalized price features
            if is_unified:
                # Normalize prices relative to the first day in the dataset
                first_close = df['Close'].iloc[0]
                df['Normalized_Close'] = df['Close'] / first_close
                df['Normalized_Open'] = df['Open'] / first_close
                df['Normalized_High'] = df['High'] / first_close
                df['Normalized_Low'] = df['Low'] / first_close
            
            # Create train/validation/test splits
            train_end = int(len(df) * 0.8)
            val_end = int(len(df) * 0.9)
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Scale the data
            scaler = MinMaxScaler()
            
            # Select features for scaling
            scale_features = self.features + ['Returns', 'MA_5', 'MA_20', 'Volatility']
            if is_unified:
                scale_features += ['Normalized_Close', 'Normalized_Open', 'Normalized_High', 'Normalized_Low']
            
            # Fit scaler on training data
            scaler.fit(train_df[scale_features])
            
            # Save the scaler
            if not is_unified:
                scaler_path = os.path.join(self.models_dir, f"{symbol}_scaler.gz")
                joblib.dump(scaler, scaler_path)
                self.scalers[symbol] = scaler
            
            # Transform all datasets
            train_scaled = pd.DataFrame(
                scaler.transform(train_df[scale_features]), 
                columns=scale_features,
                index=train_df.index
            )
            val_scaled = pd.DataFrame(
                scaler.transform(val_df[scale_features]), 
                columns=scale_features,
                index=val_df.index
            )
            test_scaled = pd.DataFrame(
                scaler.transform(test_df[scale_features]), 
                columns=scale_features,
                index=test_df.index
            )
            
            # Add date and metadata back
            train_scaled['Date'] = train_df['Date']
            val_scaled['Date'] = val_df['Date']
            test_scaled['Date'] = test_df['Date']
            
            train_scaled['Symbol'] = symbol
            val_scaled['Symbol'] = symbol
            test_scaled['Symbol'] = symbol
            
            train_scaled['Sector'] = sector
            val_scaled['Sector'] = sector
            test_scaled['Sector'] = sector
            
            # Combine into one dataframe with a split indicator
            train_scaled['Split'] = 'train'
            val_scaled['Split'] = 'val'
            test_scaled['Split'] = 'test'
            
            combined_df = pd.concat([train_scaled, val_scaled, test_scaled])
            
            return combined_df, symbol, sector
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {str(e)}")
            return None, None, None
    
    def preprocess_all_stocks(self):
        """
        Preprocess all stocks for both specific and general models
        """
        # Process individual stocks
        self.logger.info("Processing individual stock data...")
        
        # Get all sector directories
        sector_dirs = [d for d in os.listdir(self.raw_dir) 
                    if os.path.isdir(os.path.join(self.raw_dir, d)) and d != "unified"]
        
        all_stocks_data = []
        
        for sector in sector_dirs:
            sector_path = os.path.join(self.raw_dir, sector)
            stock_files = glob.glob(os.path.join(sector_path, "*_stock_price.csv"))
            
            self.logger.info(f"Processing {len(stock_files)} stocks in sector: {sector}")
            
            for file in stock_files:
                processed_df, symbol, stock_sector = self.preprocess_stock(file)
                
                if processed_df is not None:
                    # Save individual stock data
                    # Use the sector from the directory, not from the data
                    # This ensures it's always a string
                    safe_sector = str(sector).replace('/', '_')
                    output_dir = os.path.join(self.processed_dir, "specific", safe_sector)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    if symbol is not None:
                        output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
                        processed_df.to_csv(output_file, index=False)
                        
                        # Add to the list for the general model
                        all_stocks_data.append(processed_df)
                        
                        self.logger.info(f"Successfully processed {symbol}")
        
        # Combine all stocks for the general model
        if all_stocks_data:
            self.logger.info("Creating unified dataset for general model...")
            unified_df = pd.concat(all_stocks_data, ignore_index=True)
            
            # Save the unified dataset
            unified_file = os.path.join(self.processed_dir, "general", "all_stocks_processed.csv")
            unified_df.to_csv(unified_file, index=False)
            
            self.logger.info(f"Unified dataset created with {len(unified_df)} rows")
        
        self.logger.info("Preprocessing completed")

def main():
    preprocessor = SP500DataPreprocessor(
        raw_dir="data/raw",
        processed_dir="data/processed",
        models_dir="models"
    )
    preprocessor.preprocess_all_stocks()

if __name__ == "__main__":
    main()