"""
Test API routes.
"""
import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any
from datetime import datetime, timedelta

from api.routes import app
from core.config import config

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

def test_root_redirect(client):
    """Test root endpoint redirects to docs."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.endswith("/docs")

def test_api_welcome(client):
    """Test API welcome message."""
    response = client.get("/api")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()
    assert "documentation" in response.json()
    assert "endpoints" in response.json()

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "components" in data

def test_update_data(client):
    """Test data update endpoint."""
    response = client.post("/api/data/update", params={"symbol": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "stock_data" in data
    assert "news_data" in data
    assert "timestamp" in data

def test_get_stock_data(client):
    """Test stock data endpoint."""
    params = {
        "symbol": "AAPL",
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
    response = client.get("/api/data/stock", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "data" in data
    assert "columns" in data
    assert "timestamp" in data

def test_get_news_data(client):
    """Test news data endpoint."""
    response = client.get("/api/data/news", params={"symbol": "AAPL", "days": 7})
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "articles" in data
    assert "total_articles" in data
    assert "timestamp" in data

def test_list_models(client):
    """Test model listing endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total_models" in data
    assert "timestamp" in data

def test_get_model_metadata(client):
    """Test model metadata endpoint."""
    response = client.get("/api/models/AAPL/lstm")
    assert response.status_code in [200, 404]  # 404 if no model exists
    if response.status_code == 200:
        data = response.json()
        assert "symbol" in data
        assert "model_type" in data
        assert "version" in data
        assert "metadata" in data
        assert "timestamp" in data

def test_delete_model(client):
    """Test model deletion endpoint."""
    response = client.delete(
        "/api/models/AAPL/lstm",
        params={"version": "1.0.0"}
    )
    assert response.status_code in [200, 404]  # 404 if no model exists
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "message" in data

def test_invalid_symbol(client):
    """Test endpoints with invalid symbol."""
    symbol = "INVALID_SYMBOL"
    
    # Test stock data endpoint
    response = client.get("/api/data/stock", params={"symbol": symbol})
    assert response.status_code == 500
    
    # Test news data endpoint
    response = client.get("/api/data/news", params={"symbol": symbol})
    assert response.status_code == 200  # Returns empty list
    assert response.json()["total_articles"] == 0

def test_invalid_date_range(client):
    """Test stock data endpoint with invalid date range."""
    params = {
        "symbol": "AAPL",
        "start_date": datetime.now().isoformat(),
        "end_date": (datetime.now() - timedelta(days=30)).isoformat()
    }
    response = client.get("/api/data/stock", params=params)
    assert response.status_code == 500

def test_missing_parameters(client):
    """Test endpoints with missing required parameters."""
    # Test stock data endpoint without symbol
    response = client.get("/api/data/stock")
    assert response.status_code == 422
    
    # Test news data endpoint without symbol
    response = client.get("/api/data/news")
    assert response.status_code == 422
    
    # Test model deletion without version
    response = client.delete("/api/models/AAPL/lstm")
    assert response.status_code == 422

def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import asyncio
    import aiohttp
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://testserver/api/data/stock",
                params={"symbol": "AAPL"}
            ) as response:
                return await response.json()
    
    # Make 5 concurrent requests
    loop = asyncio.get_event_loop()
    tasks = [make_request() for _ in range(5)]
    responses = loop.run_until_complete(asyncio.gather(*tasks))
    
    assert len(responses) == 5
    for response in responses:
        assert "symbol" in response
        assert "data" in response 