import pytest
from unittest.mock import Mock, patch
from src.services.sentiment_service import SentimentService
from src.models.sentiment_model import SentimentModel

@pytest.fixture
def mock_sentiment_model():
    with patch('src.models.sentiment_model.SentimentModel') as mock:
        mock_model = Mock()
        mock_model.predict.return_value = {
            "sentiment": "positive",
            "confidence": 0.95,
            "scores": {
                "positive": 0.95,
                "negative": 0.02,
                "neutral": 0.03
            }
        }
        mock.return_value = mock_model
        yield mock

@pytest.fixture
def sentiment_service(mock_sentiment_model):
    return SentimentService()

@pytest.mark.asyncio
async def test_analyze_sentiment(sentiment_service):
    texts = ["This is a positive article about stocks."]
    results = await sentiment_service.analyze_sentiment(texts)
    assert len(results) == 1
    assert results[0]["sentiment"] == "positive"
    assert results[0]["confidence"] == 0.95

@pytest.mark.asyncio
async def test_analyze_sentiment_empty_text(sentiment_service):
    texts = [""]
    results = await sentiment_service.analyze_sentiment(texts)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_analyze_sentiment_multiple_texts(sentiment_service):
    texts = [
        "This is a positive article about stocks.",
        "This is a negative article about stocks."
    ]
    results = await sentiment_service.analyze_sentiment(texts)
    assert len(results) == 2
    assert all(r["sentiment"] == "positive" for r in results) 