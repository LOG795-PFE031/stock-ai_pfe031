"""
Prophet Model Trainer for Stock Price Prediction

This script trains Facebook Prophet models for stock price prediction and saves them
for later use by the API server.
"""

import os
import logging
import pandas as pd
from prophet import Prophet
from datetime import datetime
import json
from typing import Dict, Optional, List
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetModelTrainer:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed", models_dir="models/prophet"):
        """Initialize the Prophet model trainer"""
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default stock symbols if none provided
        self.default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CSCO', 'INTC']
        
    def find_stock_file(self, symbol: str) -> Optional[str]:
        """Find the stock data file in the data directories"""
        # First check in Technology sector
        tech_file = os.path.join(self.raw_dir, "Technology", f"{symbol}_stock_price.csv")
        if os.path.exists(tech_file):
            return tech_file
            
        # Then check in processed/specific directories
        for sector in os.listdir(os.path.join(self.processed_dir, "specific")):
            sector_path = os.path.join(self.processed_dir, "specific", sector)
            if os.path.isdir(sector_path):
                temp_file = os.path.join(sector_path, f"{symbol}_stock_price.csv")
                if os.path.exists(temp_file):
                    return temp_file
        
        # Finally check in raw data root
        raw_file = os.path.join(self.raw_dir, f"{symbol}_stock_price.csv")
        if os.path.exists(raw_file):
            return raw_file
            
        return None
        
    def train_model(self, symbol: str) -> Optional[Prophet]:
        """Train a Prophet model for a specific stock symbol"""
        logger.info(f"Training Prophet model for {symbol}")
        
        try:
            # Find the stock data file
            stock_file = self.find_stock_file(symbol)
            if stock_file is None:
                logger.error(f"No data found for {symbol}")
                return None
                
            logger.info(f"Found data file: {stock_file}")
            
            # Load and prepare data
            df = pd.read_csv(stock_file)
            df['ds'] = pd.to_datetime(df['Date'])
            df['y'] = df['Close']
            
            # Create and train Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add additional regressors if available
            regressors = []
            if 'Volume' in df.columns:
                df['volume'] = df['Volume'].fillna(0)
                model.add_regressor('volume')
                regressors.append('volume')
            
            if 'RSI' in df.columns:
                df['rsi'] = df['RSI'].fillna(df['RSI'].mean())
                model.add_regressor('rsi')
                regressors.append('rsi')
            
            # Fit the model
            logger.info(f"Training model on {len(df)} data points...")
            model.fit(df)
            
            # Prepare last data for saving
            last_data = df.tail(60).copy()
            last_data['ds'] = last_data['ds'].dt.strftime('%Y-%m-%d')
            
            # Save model parameters and data
            model_data = {
                'params': {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10,
                    'seasonality_mode': 'multiplicative',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': True
                },
                'regressors': regressors,
                'last_data': last_data.to_dict(orient='records'),
                'metadata': {
                    'symbol': symbol,
                    'trained_at': datetime.now().isoformat(),
                    'data_points': len(df),
                    'data_start': df['ds'].min().strftime('%Y-%m-%d'),
                    'data_end': df['ds'].max().strftime('%Y-%m-%d')
                }
            }
            
            # Save model data
            model_path = os.path.join(self.models_dir, f"{symbol}_prophet.json")
            with open(model_path, 'w') as fout:
                json.dump(model_data, fout, indent=2)
            
            logger.info(f"✅ Successfully trained and saved Prophet model for {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training Prophet model for {symbol}: {str(e)}")
            return None
    
    def train_all_models(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """Train Prophet models for all specified symbols or default ones"""
        if symbols is None:
            symbols = self.default_symbols
        
        results = {}
        for symbol in tqdm(symbols, desc="Training Prophet models"):
            model = self.train_model(symbol)
            results[symbol] = model is not None
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Training complete: {successful}/{len(symbols)} models trained successfully")
        
        # Print failed symbols if any
        failed = [symbol for symbol, success in results.items() if not success]
        if failed:
            logger.warning(f"Failed to train models for: {', '.join(failed)}")
        
        return results

def main():
    """Main function to train all Prophet models"""
    trainer = ProphetModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main() 