"""
Prediction service for stock price predictions.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from prophet import Prophet
import json
import os
from ..core.config import Config
from .rabbitmq_publisher import rabbitmq_publisher

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, specific_models, specific_scalers, general_model, general_scalers):
        self.specific_models = specific_models
        self.specific_scalers = specific_scalers
        self.general_model = general_model
        self.general_scalers = general_scalers
        self.prophet_models = {}
        self.seq_size = 60
        self.features = [
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Returns", "MA_5", "MA_20", "Volatility", "RSI", "MACD", "MACD_Signal"
        ]
        self.model_version = "1.0.0"
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_prediction(self, stock_symbol: str, model_type: str = 'lstm') -> Dict[str, Any]:
        """Get prediction for a stock using specified model type"""
        if model_type == 'prophet':
            return self._get_prophet_prediction(stock_symbol)
        else:
            return self._get_lstm_prediction(stock_symbol)

    def _get_prophet_prediction(self, stock_symbol: str) -> Dict[str, Any]:
        """Get prediction using Prophet model"""
        try:
            model = self._load_prophet_model(stock_symbol)
            stock_file = self._find_stock_file(stock_symbol)
            
            if not stock_file:
                raise FileNotFoundError(f"No data found for symbol {stock_symbol}")
            
            df = pd.read_csv(stock_file)
            df['ds'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
            df['y'] = df['Close']
            
            # Prepare regressors
            regressors = self._prepare_regressors(df)
            
            # Make prediction
            future = model.make_future_dataframe(periods=1)
            future = self._add_regressors_to_future(future, df, regressors)
            forecast = model.predict(future)
            
            # Calculate prediction and confidence
            prediction = forecast.iloc[-1]['yhat']
            confidence_score = self._calculate_prophet_confidence(forecast, df['Close'].iloc[-1])
            
            result = {
                'prediction': float(prediction),
                'timestamp': datetime.now() + timedelta(days=1),
                'confidence_score': confidence_score,
                'model_version': self.model_version,
                'model_type': 'prophet'
            }
            
            # Publish to RabbitMQ
            self._publish_prediction(stock_symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet prediction error for {stock_symbol}: {str(e)}")
            raise

    def _get_lstm_prediction(self, stock_symbol: str) -> Dict[str, Any]:
        """Get prediction using LSTM model"""
        try:
            sequence = self._get_latest_sequence(stock_symbol)
            model, scaler, model_type = self._get_model_and_scaler(stock_symbol)
            
            # Make prediction
            prediction = model.predict(sequence)
            last_sequence = sequence[0, -1, :]
            close_index = self.features.index("Close")
            last_close = last_sequence[close_index]
            
            # Get original price and prepare prediction details
            original_price = self._get_original_price(stock_symbol)
            prediction_details = self._prepare_prediction_details(
                prediction[0, 0], last_close, original_price, stock_symbol
            )
            
            # Calculate confidence and prepare result
            confidence_score = self._calculate_lstm_confidence(sequence, prediction, prediction_details)
            result = self._prepare_lstm_result(
                prediction_details['price'], confidence_score, model_type, prediction_details
            )
            
            # Publish to RabbitMQ
            self._publish_prediction(stock_symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"LSTM prediction error for {stock_symbol}: {str(e)}")
            raise

    def _load_prophet_model(self, symbol: str) -> Prophet:
        """Load or create a Prophet model for a given symbol"""
        if symbol in self.prophet_models:
            return self.prophet_models[symbol]
            
        prophet_model_path = os.path.join(self.base_dir, "models", "prophet", f"{symbol}_prophet.json")
        if os.path.exists(prophet_model_path):
            with open(prophet_model_path, 'r') as fin:
                model_data = json.load(fin)
            
            model = Prophet(**model_data['params'])
            for regressor in model_data['regressors']:
                model.add_regressor(regressor)
            
            df = pd.DataFrame(model_data['last_data'])
            df['ds'] = pd.to_datetime(df['ds'])
            model.fit(df)
            
            self.prophet_models[symbol] = model
            return model
            
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.prophet_models[symbol] = model
        return model

    def _find_stock_file(self, symbol: str) -> Optional[str]:
        """Find the stock data file in various locations"""
        processed_dir = os.path.join(self.base_dir, "data", "processed", "specific")
        raw_dir = os.path.join(self.base_dir, "data", "raw")
        
        locations = [
            os.path.join(raw_dir, "Technology", f"{symbol}_stock_price.csv"),
            *[os.path.join(processed_dir, sector, f"{symbol}_processed.csv")
              for sector in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sector))],
            os.path.join(raw_dir, f"{symbol}_stock_price.csv")
        ]
        
        for location in locations:
            if os.path.exists(location):
                return location
        return None

    def _prepare_regressors(self, df: pd.DataFrame) -> list:
        """Prepare regressors for Prophet model"""
        regressors = []
        if 'Volume' in df.columns:
            df['volume'] = df['Volume'].fillna(0)
            regressors.append('volume')
        if 'RSI' in df.columns:
            df['rsi'] = df['RSI'].fillna(df['RSI'].mean())
            regressors.append('rsi')
        return regressors

    def _add_regressors_to_future(self, future: pd.DataFrame, df: pd.DataFrame, regressors: list) -> pd.DataFrame:
        """Add regressors to future dataframe"""
        for regressor in regressors:
            future[regressor] = df[regressor].iloc[-1]
        return future

    def _calculate_prophet_confidence(self, forecast: pd.DataFrame, last_price: float) -> float:
        """Calculate confidence score for Prophet prediction"""
        lower = forecast.iloc[-1]['yhat_lower']
        upper = forecast.iloc[-1]['yhat_upper']
        interval_width = upper - lower
        return max(0, min(1, 1 - (interval_width / (4 * last_price))))

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df

    def _get_latest_sequence(self, symbol: str) -> np.ndarray:
        """Get the latest sequence for a stock"""
        stock_file = self._find_stock_file(symbol)
        if not stock_file:
            raise FileNotFoundError(f"No data found for symbol {symbol}")
            
        df = pd.read_csv(stock_file)
        df = df.tail(self.seq_size)
        
        # Calculate missing technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Fill missing values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for {symbol}: {missing_features}")
        
        return df[self.features].values.reshape(1, self.seq_size, len(self.features))

    def _get_model_and_scaler(self, symbol: str) -> tuple:
        """Get appropriate model and scaler for a symbol"""
        if symbol in self.specific_models:
            return self.specific_models[symbol], self.specific_scalers[symbol], "specific"
        elif self.general_model:
            return self.general_model, self.general_scalers['symbol'], "general"
        else:
            raise ValueError(f"No model available for {symbol}")

    def _prepare_prediction_details(self, prediction: float, last_close: float, 
                                  original_price: float, symbol: str) -> Dict[str, Any]:
        """Prepare detailed prediction information"""
        details = {
            'original_price': original_price,
            'raw_prediction': float(prediction),
            'scaling_method': 'metadata'
        }
        
        try:
            metadata_path = os.path.join(self.base_dir, "models", "specific", symbol, f"{symbol}_scaler_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                price = self._scale_with_metadata(prediction, last_close, metadata, symbol)
            else:
                price = self._simple_inverse_scale(prediction, original_price, last_close)
                details['scaling_method'] = 'simple'
            
            if original_price:
                relative_change = (price - original_price) / original_price
                details['relative_change'] = float(relative_change)
                details['change_percentage'] = float(relative_change * 100)
                
                if abs(relative_change) > 0.2:
                    conservative_price = original_price * (1 + (prediction - 1) * 0.1)
                    details['status'] = 'large_change_detected'
                    details['original_prediction'] = float(price)
                    details['conservative_estimate'] = float(conservative_price)
                    price = conservative_price
                else:
                    details['status'] = 'within_normal_range'
            
            details['price'] = price
            return details
            
        except Exception as e:
            logger.warning(f"Error in prediction scaling: {str(e)}")
            details['status'] = 'fallback_to_simple'
            details['error'] = str(e)
            details['price'] = self._simple_inverse_scale(prediction, original_price, last_close)
            return details

    def _scale_with_metadata(self, prediction: float, last_close: float, 
                           metadata: Dict[str, Any], symbol: str) -> float:
        """Scale prediction using metadata"""
        features = metadata['feature_order']
        scaler_ready = np.zeros((1, len(features)))
        
        for i, feature in enumerate(features):
            if feature == "Close":
                scaler_ready[0, i] = prediction
            elif feature in self.features:
                feat_idx = self.features.index(feature)
                scaler_ready[0, i] = last_close if feature == "Close" else 0
        
        denormalized = self.specific_scalers[symbol].inverse_transform(scaler_ready)
        return denormalized[0, features.index("Close")]

    def _simple_inverse_scale(self, predicted_normalized: float, 
                            original_price: float, last_close_normalized: float) -> float:
        """Simple scaling based on relative change"""
        if original_price is None:
            return predicted_normalized
            
        relative_change = (predicted_normalized - last_close_normalized) / last_close_normalized
        return original_price * (1 + relative_change)

    def _calculate_lstm_confidence(self, sequence: np.ndarray, prediction: np.ndarray, 
                                 details: Dict[str, Any]) -> float:
        """Calculate confidence score for LSTM prediction"""
        confidence = 0.85  # Base confidence
        
        if 'relative_change' in details:
            change_factor = abs(details['relative_change'])
            if change_factor > 0.2:
                confidence *= (1 - (change_factor - 0.2))
        
        return confidence

    def _prepare_lstm_result(self, price: float, confidence_score: float, 
                           model_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final LSTM prediction result"""
        return {
            'prediction': float(price),
            'timestamp': datetime.now() + timedelta(days=1),
            'confidence_score': confidence_score,
            'model_version': self.model_version,
            'model_type': f'lstm_{model_type}',
            'prediction_details': details
        }

    def _get_original_price(self, symbol: str) -> Optional[float]:
        """Get the last known original price for a symbol"""
        try:
            stock_file = self._find_stock_file(symbol)
            if not stock_file:
                return None
                
            df = pd.read_csv(stock_file)
            if df.empty:
                return None
                
            return float(df['Close'].iloc[-1])
            
        except Exception as e:
            logger.warning(f"Could not find original price data: {str(e)}")
            return None

    def _publish_prediction(self, symbol: str, result: Dict[str, Any]) -> None:
        """Publish prediction to RabbitMQ"""
        try:
            publish_success = rabbitmq_publisher.publish_stock_quote(symbol, result)
            if publish_success:
                logger.info(f"✅ Successfully published prediction for {symbol} to RabbitMQ")
                result['rabbitmq_status'] = 'delivered'
            else:
                logger.warning(f"⚠️ Failed to confirm RabbitMQ delivery for {symbol}")
                result['rabbitmq_status'] = 'unconfirmed'
        except Exception as e:
            logger.error(f"❌ Failed to publish to RabbitMQ: {str(e)}")
            result['rabbitmq_status'] = 'failed' 