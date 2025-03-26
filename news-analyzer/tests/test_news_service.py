import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.services.news_service import NewsService
from src.core.config import Settings

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def news_service(settings):
    return NewsService(settings)

@pytest.fixture
def mock_yfinance():
    with patch('yfinance.Ticker') as mock:
        mock_ticker = Mock()
        mock_ticker.news = [
            {
                'displayTime': '2024-03-20T10:00:00Z',
                'content': {
                    'title': 'Test Article',
                    'summary': 'Test Summary',
                    'clickThroughUrl': {'url': 'http://test.com'},
                    'provider': {'displayName': 'Test Source'}
                }
            }
        ]
        mock.return_value = mock_ticker
        yield mock

@pytest.fixture
def mock_newspaper():
    with patch('newspaper.Article') as mock:
        mock_article = Mock()
        mock_article.title = 'Test Article'
        mock_article.text = 'Test Content'
        mock.return_value = mock_article
        yield mock

@pytest.mark.asyncio
async def test_fetch_news(news_service, mock_yfinance):
    articles = await news_service.fetch_news('AAPL')
    assert len(articles) > 0
    assert articles[0]['title'] == 'Test Article'
    assert articles[0]['link'] == 'http://test.com'

@pytest.mark.asyncio
async def test_process_news(news_service, mock_newspaper):
    articles = [{
        'link': 'http://test.com',
        'title': 'Test Article',
        'date': '2024-03-20',
        'source': 'Test Source'
    }]
    processed = await news_service.process_news(articles)
    assert len(processed) > 0
    assert processed[0]['content'] == 'Test Content'

@pytest.mark.asyncio
async def test_get_news(news_service, mock_yfinance, mock_newspaper):
    articles = await news_service.get_news('AAPL')
    assert len(articles) > 0
    assert articles[0]['title'] == 'Test Article'
    assert articles[0]['content'] == 'Test Content' 