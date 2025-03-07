import unittest
import torch
from utils.sentiment_analyzer import FinBERTSentimentAnalyzer

class TestFinBERTSentimentAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the analyzer once for all tests
        cls.analyzer = FinBERTSentimentAnalyzer()
        
        # Test texts with expected sentiments
        cls.positive_text = "The company reported strong earnings, exceeding market expectations."
        cls.negative_text = "The stock plummeted after the company announced significant losses."
        cls.neutral_text = "The company announced its quarterly earnings report will be released next week."
        
    def test_analyzer_initialization(self):
        # Test that the analyzer initializes correctly
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertEqual(self.analyzer.labels, ["negative", "neutral", "positive"])
        
    def test_single_text_analysis(self):
        # Test analysis of a single text
        result = self.analyzer.analyze(self.positive_text)
        
        # Check result structure
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("scores", result)
        
        # Check scores
        self.assertIn("positive", result["scores"])
        self.assertIn("negative", result["scores"])
        self.assertIn("neutral", result["scores"])
        
        # Positive text should have higher positive score
        self.assertGreater(result["scores"]["positive"], result["scores"]["negative"])
        
    def test_batch_analysis(self):
        # Test batch analysis
        texts = [self.positive_text, self.negative_text, self.neutral_text]
        results = self.analyzer.batch_analyze(texts)
        
        # Check results
        self.assertEqual(len(results), 3)
        
        # Positive text should have positive sentiment
        self.assertGreater(results[0]["scores"]["positive"], results[0]["scores"]["negative"])
        
        # Negative text should have negative sentiment
        self.assertGreater(results[1]["scores"]["negative"], results[1]["scores"]["positive"])
        
    def test_sentiment_score(self):
        # Test normalized sentiment score
        positive_score = self.analyzer.get_sentiment_score(self.positive_text)
        negative_score = self.analyzer.get_sentiment_score(self.negative_text)
        neutral_score = self.analyzer.get_sentiment_score(self.neutral_text)
        
        # Positive text should have positive score
        self.assertGreater(positive_score, 0)
        
        # Negative text should have negative score
        self.assertLess(negative_score, 0)
        
        # Neutral text should be closer to zero than the others
        self.assertLess(abs(neutral_score), abs(positive_score))
        self.assertLess(abs(neutral_score), abs(negative_score))
        
    def test_batch_sentiment_scores(self):
        # Test batch sentiment scores
        texts = [self.positive_text, self.negative_text, self.neutral_text]
        scores = self.analyzer.get_batch_sentiment_scores(texts)
        
        # Check results
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], 0)  # Positive text
        self.assertLess(scores[1], 0)     # Negative text

if __name__ == "__main__":
    unittest.main()