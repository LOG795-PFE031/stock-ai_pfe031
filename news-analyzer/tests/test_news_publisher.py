import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.services.news_publisher import NewsPublisher

@pytest.fixture
def mock_pika():
    with patch('pika.BlockingConnection') as mock_connection, \
         patch('pika.ConnectionParameters') as mock_params, \
         patch('pika.PlainCredentials') as mock_creds:
        mock_channel = Mock()
        mock_connection.return_value.channel.return_value = mock_channel
        yield mock_connection, mock_channel

@pytest.fixture
def news_publisher(mock_pika):
    return NewsPublisher(host='test-host', exchange='test-exchange')

def test_connect_success(news_publisher, mock_pika):
    mock_connection, mock_channel = mock_pika
    assert news_publisher.connect() is True
    mock_connection.assert_called_once()
    mock_channel.exchange_declare.assert_called_once()

def test_connect_failure(news_publisher, mock_pika):
    mock_connection, _ = mock_pika
    mock_connection.side_effect = Exception("Connection failed")
    assert news_publisher.connect() is False

@pytest.mark.asyncio
async def test_publish_news_success(news_publisher, mock_pika):
    mock_connection, mock_channel = mock_pika
    result = await news_publisher.publish_news(
        title="Test Title",
        symbol="AAPL",
        content="Test Content",
        published_at=datetime.now()
    )
    assert result is True
    mock_channel.basic_publish.assert_called_once()

@pytest.mark.asyncio
async def test_publish_news_failure(news_publisher, mock_pika):
    mock_connection, mock_channel = mock_pika
    mock_channel.basic_publish.side_effect = Exception("Publish failed")
    result = await news_publisher.publish_news(
        title="Test Title",
        symbol="AAPL",
        content="Test Content"
    )
    assert result is False

def test_cleanup(news_publisher, mock_pika):
    mock_connection, mock_channel = mock_pika
    news_publisher._cleanup()
    mock_channel.close.assert_called_once()
    mock_connection.close.assert_called_once()

def test_cleanup_on_exit(news_publisher, mock_pika):
    mock_connection, mock_channel = mock_pika
    news_publisher._cleanup_on_exit()
    assert news_publisher._is_shutting_down is True
    mock_channel.close.assert_called_once()
    mock_connection.close.assert_called_once() 