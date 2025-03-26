import pytest
from unittest.mock import Mock, patch
from fastapi import Request
from src.middleware.rate_limit import RateLimitMiddleware
from src.core.config import Settings

@pytest.fixture
def settings():
    return Settings(RATE_LIMIT=2, RATE_LIMIT_PERIOD=1)

@pytest.fixture
def middleware(settings):
    return RateLimitMiddleware(settings)

@pytest.fixture
def mock_request():
    request = Mock(spec=Request)
    request.client.host = "127.0.0.1"
    return request

@pytest.mark.asyncio
async def test_rate_limit_not_exceeded(middleware, mock_request):
    # First request should pass
    response = await middleware(mock_request, lambda: Mock(status_code=200))
    assert response.status_code == 200

    # Second request should pass
    response = await middleware(mock_request, lambda: Mock(status_code=200))
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_rate_limit_exceeded(middleware, mock_request):
    # Make requests up to the limit
    for _ in range(2):
        await middleware(mock_request, lambda: Mock(status_code=200))

    # Next request should be rate limited
    with pytest.raises(Exception) as exc_info:
        await middleware(mock_request, lambda: Mock(status_code=200))
    assert exc_info.value.status_code == 429

@pytest.mark.asyncio
async def test_rate_limit_reset(middleware, mock_request):
    # Make requests up to the limit
    for _ in range(2):
        await middleware(mock_request, lambda: Mock(status_code=200))

    # Wait for rate limit period to reset
    import time
    time.sleep(1.1)  # Wait slightly longer than the period

    # Request should pass after reset
    response = await middleware(mock_request, lambda: Mock(status_code=200))
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_different_ips_not_limited(middleware):
    # Create requests from different IPs
    request1 = Mock(spec=Request)
    request1.client.host = "127.0.0.1"
    request2 = Mock(spec=Request)
    request2.client.host = "127.0.0.2"

    # Both requests should pass
    response1 = await middleware(request1, lambda: Mock(status_code=200))
    response2 = await middleware(request2, lambda: Mock(status_code=200))
    assert response1.status_code == 200
    assert response2.status_code == 200 