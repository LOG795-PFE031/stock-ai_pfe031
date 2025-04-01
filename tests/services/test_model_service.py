"""
Tests for the model service.
"""
import pytest
from datetime import datetime
import json
from pathlib import Path

pytestmark = pytest.mark.asyncio

async def test_model_service_initialization(model_service, config):
    """Test model service initialization."""
    assert model_service._initialized
    assert Path(config.model.PREDICTION_MODELS_DIR).exists()
    assert Path(config.model.NEWS_MODELS_DIR).exists()

async def test_save_lstm_model(model_service, config, sample_lstm_model):
    """Test saving LSTM model."""
    symbol = "AAPL"
    model_type = "lstm"
    metrics = {"mse": 0.001, "mae": 0.02}
    metadata = {"epochs": 50, "batch_size": 32}
    
    result = await model_service.save_model(
        symbol,
        model_type,
        sample_lstm_model,
        metrics,
        metadata
    )
    
    assert isinstance(result, dict)
    assert result["symbol"] == symbol
    assert result["model_type"] == model_type
    assert result["version"] == model_service.model_version
    assert result["metrics"] == metrics
    
    # Check if files are created
    version_dir = model_service._get_version_dir(symbol, model_type)
    assert version_dir.exists()
    assert (version_dir / f"{symbol}_{model_type}_model").exists()
    assert (version_dir / f"{symbol}_{model_type}_metrics.json").exists()
    assert (version_dir / f"{symbol}_{model_type}_metadata.json").exists()

async def test_save_prophet_model(model_service, config, sample_prophet_model):
    """Test saving Prophet model."""
    symbol = "AAPL"
    model_type = "prophet"
    metrics = {"rmse": 1.5, "mape": 0.05}
    metadata = {"changepoint_prior_scale": 0.05}
    
    result = await model_service.save_model(
        symbol,
        model_type,
        sample_prophet_model,
        metrics,
        metadata
    )
    
    assert isinstance(result, dict)
    assert result["symbol"] == symbol
    assert result["model_type"] == model_type
    assert result["metrics"] == metrics
    
    # Check if files are created
    version_dir = model_service._get_version_dir(symbol, model_type)
    assert version_dir.exists()
    assert (version_dir / f"{symbol}_{model_type}_model.joblib").exists()

async def test_load_model(model_service, config, sample_lstm_model):
    """Test loading a saved model."""
    symbol = "AAPL"
    model_type = "lstm"
    metrics = {"mse": 0.001}
    
    # Save model first
    await model_service.save_model(symbol, model_type, sample_lstm_model, metrics)
    
    # Load model
    result = await model_service.load_model(symbol, model_type)
    
    assert isinstance(result, dict)
    assert "model" in result
    assert "metadata" in result
    assert "version" in result
    assert result["version"] == model_service.model_version

async def test_list_models(model_service, sample_lstm_model, sample_prophet_model):
    """Test listing available models."""
    # Save multiple models
    await model_service.save_model("AAPL", "lstm", sample_lstm_model, {"mse": 0.001})
    await model_service.save_model("GOOGL", "lstm", sample_lstm_model, {"mse": 0.002})
    await model_service.save_model("AAPL", "prophet", sample_prophet_model, {"rmse": 1.5})
    
    # List all models
    models = await model_service.list_models()
    assert len(models) == 3
    
    # List models by symbol
    aapl_models = await model_service.list_models(symbol="AAPL")
    assert len(aapl_models) == 2
    
    # List models by type
    lstm_models = await model_service.list_models(model_type="lstm")
    assert len(lstm_models) == 2

async def test_delete_model(model_service, sample_lstm_model):
    """Test deleting a model."""
    symbol = "AAPL"
    model_type = "lstm"
    
    # Save model first
    await model_service.save_model(symbol, model_type, sample_lstm_model, {"mse": 0.001})
    
    # Delete model
    success = await model_service.delete_model(
        symbol,
        model_type,
        model_service.model_version
    )
    assert success
    
    # Try to load deleted model
    with pytest.raises(ValueError):
        await model_service.load_model(symbol, model_type)

async def test_load_nonexistent_model(model_service):
    """Test loading a model that doesn't exist."""
    with pytest.raises(ValueError):
        await model_service.load_model("NONEXISTENT", "lstm")

async def test_delete_nonexistent_model(model_service):
    """Test deleting a model that doesn't exist."""
    success = await model_service.delete_model("NONEXISTENT", "lstm", "1.0.0")
    assert not success

async def test_model_metadata_persistence(model_service, config, sample_lstm_model):
    """Test that model metadata persists between service instances."""
    symbol = "AAPL"
    model_type = "lstm"
    metrics = {"mse": 0.001}
    
    # Save model with first service instance
    await model_service.save_model(symbol, model_type, sample_lstm_model, metrics)
    
    # Create new service instance
    new_service = model_service.__class__()
    await new_service.initialize()
    
    # Check if metadata is loaded
    models = await new_service.list_models()
    assert len(models) > 0
    assert any(m["symbol"] == symbol and m["model_type"] == model_type for m in models)
    
    await new_service.cleanup()

async def test_model_service_cleanup(model_service):
    """Test model service cleanup."""
    await model_service.cleanup()
    assert not model_service._initialized 