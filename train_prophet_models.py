"""
Script to train Prophet models for all symbols that have LSTM models.
"""

import asyncio
import os
from datetime import datetime, timedelta
from services.model_service import ModelService
from services.data_service import DataService
from services import TrainingService
from core.logging import logger


async def train_prophet_models():
    """Train Prophet models for all symbols that have LSTM models."""
    model_service = None
    data_service = None
    training_service = None

    try:
        # Initialize services
        model_service = ModelService()
        data_service = DataService()

        # Initialize services in order
        await model_service.initialize()
        await data_service.initialize()

        # Create training service with dependencies
        training_service = TrainingService(
            model_service=model_service, data_service=data_service
        )
        await training_service.initialize()

        # Get all symbols with LSTM models
        symbols = list(model_service._specific_models.keys())
        logger["training"].info(f"Found {len(symbols)} symbols with LSTM models")

        # Train Prophet models for each symbol
        for symbol in symbols:
            try:
                logger["training"].info(f"Training Prophet model for {symbol}")
                result = await training_service.train_model(
                    symbol=symbol,
                    model_type="prophet",
                    start_date=datetime.now()
                    - timedelta(days=365 * 2),  # 2 years of data
                    end_date=datetime.now(),
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    holidays_prior_scale=10.0,
                    seasonality_mode="multiplicative",
                )

                if result.get("status") == "success":
                    logger["training"].info(
                        f"Successfully trained Prophet model for {symbol}"
                    )
                    logger["training"].info(
                        f"Model metrics: {result.get('result', {}).get('metrics', {})}"
                    )
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger["training"].error(
                        f"Failed to train Prophet model for {symbol}: {error_msg}"
                    )
            except Exception as e:
                logger["training"].error(
                    f"Error training Prophet model for {symbol}: {str(e)}"
                )
                continue

        logger["training"].info("Prophet model training completed")

    except Exception as e:
        logger["training"].error(f"Error in train_prophet_models: {str(e)}")
    finally:
        # Cleanup services in reverse order
        try:
            if training_service:
                await training_service.cleanup()
            if data_service:
                await data_service.cleanup()
            if model_service:
                await model_service.cleanup()
        except Exception as e:
            logger["training"].error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    asyncio.run(train_prophet_models())
