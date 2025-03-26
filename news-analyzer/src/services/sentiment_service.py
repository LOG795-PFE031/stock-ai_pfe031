from typing import List, Dict
import logging
from ..models.sentiment_model import SentimentModel

logger = logging.getLogger(__name__)

class SentimentService:
    def __init__(self):
        self.model = SentimentModel()
        logger.info("SentimentService initialized")

    async def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a list of texts"""
        try:
            results = []
            for text in texts:
                if not text or not text.strip():
                    continue
                result = await self.model.predict(text)
                
                # Convert raw sentiment to opinion score
                sentiment_map = {
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0
                }
                opinion = sentiment_map.get(result["sentiment"], 0)
                
                # Create detailed sentiment result
                sentiment_result = {
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "scores": result["scores"],
                    "opinion": opinion,
                    "summary": self._get_sentiment_summary(result["sentiment"], result["confidence"])
                }
                results.append(sentiment_result)
            return results
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return [{
                "sentiment": "neutral",
                "confidence": 1.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "opinion": 0,
                "summary": "Unable to analyze sentiment"
            } for _ in texts]

    def _get_sentiment_summary(self, sentiment: str, confidence: float) -> str:
        """Generate a human-readable summary of the sentiment analysis"""
        confidence_percentage = round(confidence * 100)
        if confidence_percentage >= 80:
            strength = "strongly"
        elif confidence_percentage >= 60:
            strength = "moderately"
        else:
            strength = "slightly"
            
        return f"{strength.capitalize()} {sentiment} sentiment ({confidence_percentage}% confidence)" 