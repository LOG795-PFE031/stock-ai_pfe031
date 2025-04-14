"""
Tests for the prediction service.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
from pathlib import Path

from services.prediction_service import PredictionService
from services.model_service import ModelService
from services.data_service import DataService
from services.rabbitmq_service import RabbitMQService
from core.logging import logger

@pytest.fixture
def mock_model_service():
    """Create a mock model service."""
    service = Mock(spec=ModelService)
    service._specific_models = {"AAPL": "mock_model", "MSFT": "mock_model"}
    service.prophet_dir = Path("/app/models/prophet")
    service.prophet_dir.glob = Mock(return_value=[
        Path("AAPL_prophet.json"),
        Path("MSFT_prophet.json")
    ])
    return service

@pytest.fixture
def mock_data_service():
    """Create a mock data service."""
    service = Mock(spec=DataService)
    service.get_latest_data = AsyncMock(return_value={
        "status": "success",
        "data": {
            "Date": [datetime.now()],
            "Close": [150.0],
            "Open": [149.0],
            "High": [151.0],
            "Low": [148.0],
            "Volume": [1000000]
        }
    })
    return service

@pytest.fixture
def mock_rabbitmq_service():
    """Create a mock RabbitMQ service."""
    service = Mock(spec=RabbitMQService)
    service.publish_stock_quote = Mock(return_value=True)
    return service

@pytest.mark.asyncio
async def test_day_started_event_handling(mock_model_service, mock_data_service, mock_rabbitmq_service):
    """Test that day-started events trigger prediction publishing."""
    # Create prediction service with mocks
    service = PredictionService(mock_model_service, mock_data_service)
    service.rabbitmq_service = mock_rabbitmq_service
    
    # Initialize the service
    await service.initialize()
    
    # Simulate a day-started event
    day_started_message = {
        "event_type": "DayStarted",
        "timestamp": datetime.now().isoformat(),
        "day": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Call the day-started callback directly
    service._on_day_started(day_started_message)
    
    # Wait for predictions to be published
    await asyncio.sleep(1)  # Give time for async tasks to complete
    
    # Verify that predictions were published for each model
    assert mock_rabbitmq_service.publish_stock_quote.call_count >= 4  # 2 symbols * 2 model types
    
    # Verify the message format
    for call in mock_rabbitmq_service.publish_stock_quote.call_args_list:
        symbol, prediction = call.args
        assert isinstance(symbol, str)
        assert isinstance(prediction, dict)
        assert "prediction" in prediction
        assert "confidence_score" in prediction
        assert "model_type" in prediction
        assert "model_version" in prediction
        assert "timestamp" in prediction

@pytest.mark.asyncio
async def test_prediction_service_cleanup(mock_model_service, mock_data_service, mock_rabbitmq_service):
    """Test that the service cleans up properly."""
    service = PredictionService(mock_model_service, mock_data_service)
    service.rabbitmq_service = mock_rabbitmq_service
    
    # Initialize and start auto-publishing
    await service.initialize()
    await service.start_auto_publishing(interval_minutes=1)
    
    # Clean up
    await service.cleanup()
    
    # Verify cleanup was called
    assert service._stop_publishing is True
    assert service._publish_task is None
    mock_rabbitmq_service.close.assert_called_once()

@pytest.mark.asyncio
async def test_prediction_service_error_handling(mock_model_service, mock_data_service, mock_rabbitmq_service):
    """Test error handling in the prediction service."""
    service = PredictionService(mock_model_service, mock_data_service)
    service.rabbitmq_service = mock_rabbitmq_service
    
    # Make data service raise an error
    mock_data_service.get_latest_data = AsyncMock(side_effect=Exception("Test error"))
    
    # Initialize the service
    await service.initialize()
    
    # Simulate a day-started event
    day_started_message = {
        "event_type": "DayStarted",
        "timestamp": datetime.now().isoformat(),
        "day": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Call the day-started callback directly
    service._on_day_started(day_started_message)
    
    # Wait for predictions to be published
    await asyncio.sleep(1)
    
    # Verify that the service handled the error gracefully
    assert mock_rabbitmq_service.publish_stock_quote.call_count == 0  # No successful publishes 