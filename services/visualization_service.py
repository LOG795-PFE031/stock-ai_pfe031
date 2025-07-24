"""
Stock visualization service using Plotly.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime

from core import BaseService
from services import DataService
from core.logging import logger


class VisualizationService(BaseService):
    """Service for generating interactive stock visualizations."""

    def __init__(self, data_service: DataService):
        super().__init__()
        self.data_service = data_service
        self.logger = logger["visualization"]

    def initialize(self):
        return super().initialize()

    def cleanup(self):
        return super().cleanup()

    async def get_stock_chart(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate an interactive stock chart with technical indicators.

        Args:
            symbol: Stock symbol
            days: Number of days of historical data to include
            include_indicators: Whether to include technical indicators

        Returns:
            Dictionary containing the Plotly figure and metadata
        """
        try:
            # Get historical data
            df, symbol_name = await self.data_service.get_recent_data(
                symbol, days_back=days
            )

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                    )
                ]
            )

            # Update layout
            fig.update_layout(
                title=f"{symbol_name} ({symbol}) Stock Price",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=800,
                template="plotly_dark",
            )

            # If we want a quick look at the chart
            # fig.show()

            fig2 = px.line(df, x="Date", y="Close")
            fig2.show()

            return {
                "status": "success",
                "figure": fig.to_json(),
                "metadata": {
                    "symbol": symbol,
                    "last_updated": datetime.now().isoformat(),
                    "data_points": len(df),
                    "date_range": {
                        "start": df["Date"].min().isoformat(),
                        "end": df["Date"].max().isoformat(),
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating stock chart for {symbol}: {str(e)}")
            print(str(e))
            return {"status": "error", "error": str(e)}

    async def get_prediction_chart(
        self, symbol: str, days: int = 30, model_type: str = "lstm"
    ) -> Dict[str, Any]:
        """
        Generate an interactive chart showing historical data and predictions.

        Args:
            symbol: Stock symbol
            days: Number of days of historical data to include
            model_type: Type of model to use for prediction

        Returns:
            Dictionary containing the Plotly figure and metadata
        """
        try:
            # Get historical data
            data_result = await self.data_service.get_historical_data(symbol, days=days)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")

            df = data_result["data"]

            # Get predictions
            from services.prediction_service import PredictionService

            prediction_service = PredictionService(None, self.data_service)
            prediction_result = await prediction_service.get_historical_predictions(
                symbol, days=days
            )

            if prediction_result["status"] != "success":
                raise RuntimeError(f"Failed to get predictions for {symbol}")

            predictions = prediction_result["historical_predictions"]
            pred_df = pd.DataFrame(predictions)

            # Create figure
            fig = go.Figure()

            # Add actual price
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["Close"],
                    name="Actual Price",
                    line=dict(color="blue"),
                )
            )

            # Add predictions
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(pred_df["date"]),
                    y=pred_df["prediction"],
                    name="Predicted Price",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Add error bars
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(pred_df["date"]),
                    y=pred_df["error"],
                    name="Prediction Error",
                    line=dict(color="gray"),
                    opacity=0.5,
                )
            )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Predictions vs Actual",
                yaxis_title="Price",
                xaxis_title="Date",
                height=600,
                template="plotly_dark",
            )

            return {
                "status": "success",
                "figure": fig.to_json(),
                "metadata": {
                    "symbol": symbol,
                    "model_type": model_type,
                    "last_updated": datetime.now().isoformat(),
                    "data_points": len(df),
                    "prediction_points": len(pred_df),
                    "date_range": {
                        "start": df["Date"].min().isoformat(),
                        "end": df["Date"].max().isoformat(),
                    },
                },
            }

        except Exception as e:
            self.logger.error(
                f"Error generating prediction chart for {symbol}: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    async def get_correlation_matrix(
        self, symbols: List[str], days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate a correlation matrix heatmap for multiple stocks.

        Args:
            symbols: List of stock symbols
            days: Number of days of historical data to include

        Returns:
            Dictionary containing the Plotly figure and metadata
        """
        try:
            # Get data for all symbols
            all_data = {}
            for symbol in symbols:
                data_result = await self.data_service.get_historical_data(
                    symbol, days=days
                )
                if data_result["status"] == "success":
                    all_data[symbol] = data_result["data"]

            if not all_data:
                raise RuntimeError("No data available for any of the symbols")

            # Calculate correlations
            correlations = pd.DataFrame()
            for symbol, df in all_data.items():
                correlations[symbol] = df["Close"].pct_change()

            correlation_matrix = correlations.corr()

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                )
            )

            # Update layout
            fig.update_layout(
                title="Stock Correlation Matrix",
                xaxis_title="Symbol",
                yaxis_title="Symbol",
                height=600,
                template="plotly_dark",
            )

            return {
                "status": "success",
                "figure": fig.to_json(),
                "metadata": {
                    "symbols": symbols,
                    "last_updated": datetime.now().isoformat(),
                    "days": days,
                    "correlation_range": {
                        "min": correlation_matrix.min().min(),
                        "max": correlation_matrix.max().max(),
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating correlation matrix: {str(e)}")
            print(str(e))
            return {"status": "error", "error": str(e)}
