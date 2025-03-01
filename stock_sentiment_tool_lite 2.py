#!/usr/bin/env python
# coding: utf-8

"""
Stock Sentiment Analysis Tool (Lite Version)

This module analyzes sentiment for a given stock ticker using mock data
and simulated sentiment analysis results.
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Union
import re


class MockSentimentAnalyzer:
    """
    A mock sentiment analyzer that returns simulated sentiment scores.
    """
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Generate mock sentiment analysis results.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of dictionaries with mock sentiment results
        """
        results = []
        sentiments = ["positive", "negative", "neutral"]
        
        for text in texts:
            # Simple heuristic: check for positive/negative keywords
            positive_words = ["bullish", "rose", "strong", "beats", "up", "buy", "moon", "crushing"]
            negative_words = ["bearish", "scrutiny", "concerns", "worried", "down", "sell", "lost", "disappointed"]
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word.lower() in text.lower())
            neg_count = sum(1 for word in negative_words if word.lower() in text.lower())
            
            # Determine sentiment based on keyword count
            if pos_count > neg_count:
                sentiment = "positive"
                pos_score = random.uniform(0.6, 0.9)
                neg_score = random.uniform(0.0, 0.2)
                neu_score = 1.0 - pos_score - neg_score
            elif neg_count > pos_count:
                sentiment = "negative"
                neg_score = random.uniform(0.6, 0.9)
                pos_score = random.uniform(0.0, 0.2)
                neu_score = 1.0 - pos_score - neg_score
            else:
                sentiment = "neutral"
                neu_score = random.uniform(0.6, 0.9)
                pos_score = random.uniform(0.0, 0.2)
                neg_score = 1.0 - pos_score - neu_score
                
            # Generate confidence value
            confidence = random.uniform(0.7, 0.95)
            
            results.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": {
                    "positive": pos_score,
                    "negative": neg_score,
                    "neutral": neu_score
                }
            })
            
        return results


class StockSentimentTool:
    """
    A tool for analyzing sentiment related to specific stocks based on
    news and social media data.
    """
    
    def __init__(self):
        """Initialize the Stock Sentiment Analysis Tool."""
        self.analyzer = MockSentimentAnalyzer()
        self.sources = ["WallStreet Journal", "YahooFinance", "Reddit", "Twitter"]
        
    def get_mock_data(self, ticker: str) -> Dict[str, List[str]]:
        """
        Generate mock text data for a given stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict[str, List[str]]: Dictionary with source names as keys and lists of texts as values
        """
        # Clean and standardize ticker
        ticker = ticker.strip().upper()
        
        # Dictionary to store full company names for common tickers
        company_names = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA"
        }
        
        # Get company name if available, otherwise use ticker
        company = company_names.get(ticker, ticker)
        
        # Mock data templates for each source
        wsj_templates = [
            f"{company} shares rose today after the company reported strong quarterly results.",
            f"Analysts are bullish on {company}'s new product lineup, raising price targets.",
            f"{company} (${ticker}) faces regulatory scrutiny over recent business practices.",
            f"Market concerns grow as {company} delays its upcoming product launch.",
            f"{company} announced a new CEO today, signaling a strategic shift."
        ]
        
        yahoo_templates = [
            f"{company} beats earnings expectations, stock jumps in after-hours trading.",
            f"Why {company} (${ticker}) could be a strong buy according to analysts.",
            f"Investors worried about {company}'s growth slowdown in key markets.",
            f"{company} announces share buyback program worth billions.",
            f"Is {company} overvalued? Experts weigh in on the tech giant's prospects."
        ]
        
        reddit_templates = [
            f"Just bought more {ticker}! This company is going to the moon! 🚀🚀🚀",
            f"Should I sell my {ticker} shares before earnings? Looking for advice.",
            f"DD on {company}: Why I think it's undervalued right now.",
            f"Not happy with my {ticker} investment. The management seems lost.",
            f"What's everyone's price target for {ticker} by end of year?"
        ]
        
        twitter_templates = [
            f"#{ticker} crushing it today! Best stock in my portfolio right now.",
            f"Disappointed by ${ticker}'s latest product announcement. Might be time to sell.",
            f"Just read that {company} is expanding into new markets. Bullish! #{ticker}",
            f"${ticker} down 3% on no news? Time to buy the dip!",
            f"Anyone else concerned about {company}'s debt levels? #{ticker} #investing"
        ]
        
        # Randomly select 2-3 texts from each template list
        mock_data = {
            "WallStreet Journal": random.sample(wsj_templates, random.randint(2, min(3, len(wsj_templates)))),
            "YahooFinance": random.sample(yahoo_templates, random.randint(2, min(3, len(yahoo_templates)))),
            "Reddit": random.sample(reddit_templates, random.randint(2, min(3, len(reddit_templates)))),
            "Twitter": random.sample(twitter_templates, random.randint(2, min(3, len(twitter_templates))))
        }
        
        return mock_data
    
    def calculate_relevance(self, texts: List[str], ticker: str) -> float:
        """
        Calculate relevance score based on ticker mentions and context.
        
        Args:
            texts (List[str]): List of texts to analyze
            ticker (str): Stock ticker to calculate relevance for
            
        Returns:
            float: Relevance score between 0 and 1
        """
        if not texts:
            return 0.0
            
        # Clean ticker for comparison
        ticker = ticker.strip().upper()
        
        # Initialize counters
        total_texts = len(texts)
        ticker_mentions = 0
        ticker_patterns = [
            re.compile(r'\b' + re.escape(ticker) + r'\b', re.IGNORECASE),  # Exact ticker match
            re.compile(r'\$' + re.escape(ticker) + r'\b', re.IGNORECASE),  # $TICKER format
            re.compile(r'#' + re.escape(ticker) + r'\b', re.IGNORECASE)    # #TICKER format
        ]
        
        for text in texts:
            # Check for ticker mentions
            for pattern in ticker_patterns:
                if pattern.search(text):
                    ticker_mentions += 1
                    break
        
        # Calculate basic relevance score based on percentage of texts mentioning the ticker
        relevance_score = ticker_mentions / total_texts if total_texts > 0 else 0
        
        # Add some randomness to simulate more complex relevance algorithms
        relevance_score = min(1.0, relevance_score + random.uniform(-0.1, 0.2))
        
        return max(0.0, round(relevance_score, 2))
    
    def analyze_sentiment(self, ticker: str) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Analyze sentiment for a given stock ticker using mock data.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict: Dictionary containing sentiment analysis results
        """
        # Get mock data for the ticker
        mock_data = self.get_mock_data(ticker)
        
        # Flatten all texts for overall analysis
        all_texts = []
        for source_texts in mock_data.values():
            all_texts.extend(source_texts)
        
        # Calculate relevance score
        relevance = self.calculate_relevance(all_texts, ticker)
        
        # Analyze sentiment for all texts
        sentiment_results = self.analyzer.batch_analyze(all_texts)
        
        # Calculate overall sentiment scores
        total_positive = sum(result['scores']['positive'] for result in sentiment_results)
        total_negative = sum(result['scores']['negative'] for result in sentiment_results)
        total_neutral = sum(result['scores']['neutral'] for result in sentiment_results)
        
        # Determine overall sentiment
        if total_positive > total_negative and total_positive > total_neutral:
            overall_sentiment = "positive"
        elif total_negative > total_positive and total_negative > total_neutral:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
            
        # Prepare source-specific results
        source_results = []
        text_index = 0
        for source, texts in mock_data.items():
            source_sentiment_results = sentiment_results[text_index:text_index + len(texts)]
            text_index += len(texts)
            
            # Calculate source-specific sentiment
            source_positive = sum(result['scores']['positive'] for result in source_sentiment_results)
            source_negative = sum(result['scores']['negative'] for result in source_sentiment_results)
            source_neutral = sum(result['scores']['neutral'] for result in source_sentiment_results)
            
            if source_positive > source_negative and source_positive > source_neutral:
                source_overall_sentiment = "positive"
            elif source_negative > source_positive and source_negative > source_neutral:
                source_overall_sentiment = "negative"
            else:
                source_overall_sentiment = "neutral"
                
            source_results.append({
                "source": source,
                "sentiment": source_overall_sentiment,
                "text_count": len(texts),
                "relevance": self.calculate_relevance(texts, ticker)
            })
        
        # Compile final results
        return {
            "ticker": ticker,
            "sentiment": overall_sentiment,
            "relevance": relevance,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source_breakdown": source_results
        }
    
    def get_sentiment_json(self, ticker: str) -> str:
        """
        Get sentiment analysis results in JSON format.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: JSON string with analysis results
        """
        results = self.analyze_sentiment(ticker)
        
        # Extract only the required fields for the simple output format
        simple_output = {
            "sentiment": results["sentiment"],
            "relevance": results["relevance"],
            "date": results["date"]
        }
        
        return json.dumps(simple_output, indent=2)


def main():
    """
    Main function to demonstrate the stock sentiment analysis tool.
    """
    # Default ticker
    default_ticker = "AAPL"
    
    # Create the sentiment analysis tool
    sentiment_tool = StockSentimentTool()
    
    # Analyze sentiment for the default ticker
    print(f"\nAnalyzing sentiment for {default_ticker}...\n")
    
    # Get sentiment results
    results_json = sentiment_tool.get_sentiment_json(default_ticker)
    
    # Print results
    print(results_json)
    
    # Get detailed results for demonstration
    detailed_results = sentiment_tool.analyze_sentiment(default_ticker)
    
    # Print source breakdown
    print("\nSource Breakdown:")
    for source in detailed_results["source_breakdown"]:
        print(f"- {source['source']}: {source['sentiment']} (Relevance: {source['relevance']})")
    
    # Show a sample of the mock texts
    print("\nSample Texts:")
    mock_data = sentiment_tool.get_mock_data(default_ticker)
    for source, texts in mock_data.items():
        print(f"\n{source}:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")


if __name__ == "__main__":
    main()