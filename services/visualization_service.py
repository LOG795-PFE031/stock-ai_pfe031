"""
Stock visualization service using Plotly.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

from .base_service import BaseService
from .data_service import DataService
from core.logging import logger


class VisualizationService(BaseService):
    """Service for generating interactive stock visualizations."""

    def __init__(self, data_service: DataService):
        super().__init__()
        self.data_service = data_service
        self.logger = logger["visualization"]

    async def get_stock_chart(
        self, symbol: str, days: int = 30, include_indicators: bool = True
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
            data_result = await self.data_service.get_historical_data(symbol, days=days)
            if data_result["status"] != "success":
                raise RuntimeError(f"Failed to get historical data for {symbol}")

            df = data_result["data"]

            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2 if include_indicators else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3] if include_indicators else [1.0],
            )

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                ),
                row=1,
                col=1,
            )

            # Add volume
            fig.add_trace(
                go.Bar(x=df["Date"], y=df["Volume"], name="Volume", opacity=0.5),
                row=1,
                col=1,
            )

            if include_indicators:
                # Add technical indicators
                if "RSI" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["RSI"],
                            name="RSI",
                            line=dict(color="purple"),
                        ),
                        row=2,
                        col=1,
                    )
                    # Add RSI overbought/oversold lines
                    fig.add_hline(
                        y=70, line_dash="dash", line_color="red", row=2, col=1
                    )
                    fig.add_hline(
                        y=30, line_dash="dash", line_color="green", row=2, col=1
                    )

                if "MACD" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["MACD"],
                            name="MACD",
                            line=dict(color="blue"),
                        ),
                        row=2,
                        col=1,
                    )
                    if "MACD_Signal" in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df["Date"],
                                y=df["MACD_Signal"],
                                name="MACD Signal",
                                line=dict(color="orange"),
                            ),
                            row=2,
                            col=1,
                        )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Stock Price and Indicators",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=800,
                template="plotly_dark",
            )

            if include_indicators:
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Indicators", row=2, col=1)

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
            return {"status": "error", "error": str(e)}
