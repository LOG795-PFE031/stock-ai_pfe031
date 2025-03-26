from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class SentimentModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Device selection for GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0
            logger.info("CUDA available, using GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1
            logger.info("MPS available, using Apple Silicon GPU")
        else:
            self.device = torch.device("cpu")
            device_id = -1
            logger.info("No GPU available, using CPU")

        logger.info("Loading FinBERT model...")
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model.to(self.device)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_id)
        logger.info("FinBERT model loaded successfully")
        self._initialized = True

    async def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        try:
            # BERT models have a maximum token limit of 512 tokens
            max_length = 512
            
            # Tokenize with truncation
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Send to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get probabilities and sentiment
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
            # Map indices to sentiments
            sentiment_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
            sentiment_idx = torch.argmax(probs).item()
            sentiment = sentiment_map[sentiment_idx]
            confidence = probs[sentiment_idx].item()
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": {
                    "positive": probs[1].item(),
                    "negative": probs[2].item(),
                    "neutral": probs[0].item()
                }
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 1.0, "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} 