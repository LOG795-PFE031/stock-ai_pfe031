#!/usr/bin/env python
# coding: utf-8

"""
Stock Sentiment Analysis Tool (Real Data Version)

This module analyzes sentiment for a given stock ticker or name by processing
real-time text data from financial news sources using the FinBERT model.
"""

import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import sentiment analyzer
try:
    from utils.sentiment_analyzer import FinBERTSentimentAnalyzer
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False
    logger.warning("FinBERT model not available. Using mock sentiment results.")

# Import web scraper
try:
    from utils.webscraper_real import WebScraper
    HAS_WEBSCRAPER = True
except ImportError:
    HAS_WEBSCRAPER = False
    logger.warning("Web scraper not available. Using mock data.")


class MockSentimentAnalyzer:
    """
    A mock sentiment analyzer that returns simulated sentiment scores.
    Used as a fallback when the actual model can't be loaded.
    """
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Generate mock sentiment analysis results.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of dictionaries with mock sentiment results
        """
        import random
        results = []
        
        for text in texts:
            # Simple heuristic: check for positive/negative keywords
            positive_words = ["bullish", "rose", "strong", "beats", "up", "buy", "moon", "crushing", 
                             "positive", "growth", "gain", "profit", "success", "improve", "boost"]
            negative_words = ["bearish", "scrutiny", "concerns", "worried", "down", "sell", "lost", "disappointed",
                             "negative", "decline", "loss", "risk", "trouble", "drop", "fall"]
            
            # Count positive and negative words
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
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
                
            results.append({
                "sentiment": sentiment,
                "confidence": random.uniform(0.7, 0.95),
                "scores": {
                    "positive": pos_score,
                    "negative": neg_score,
                    "neutral": neu_score
                }
            })
            
        return results


def generate_mock_data(ticker: str) -> Dict[str, List[str]]:
    """
    Generate mock data when real scraping fails.
    
    Args:
        ticker (str): Stock ticker
        
    Returns:
        Dict[str, List[str]]: Mock data
    """
    logger.warning(f"Generating mock data for {ticker} as fallback")
    
    # Get company name
    ticker_to_company = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA"
    }
    company = ticker_to_company.get(ticker, ticker)
    
    # Generate some mock headlines
    google_news = [
        f"{company} stock rises after analyst upgrade",
        f"{ticker} beats earnings expectations, shares jump",
        f"Analysts positive on {company}'s growth prospects",
        f"{company} announces new product line, stock reacts",
        f"Market awaits {company}'s quarterly results"
    ]
    
    yahoo_headlines = [
        f"{company} hits new 52-week high",
        f"Why {ticker} could be a good buy right now",
        f"{company}'s CEO discusses future growth in interview",
        f"3 reasons to be bullish on {ticker}",
        f"Analysts set new price targets for {company}"
    ]
    
    market_watch = [
        f"Opinion: {company} is positioned for long-term growth",
        f"{ticker} shares react to market volatility",
        f"Earnings preview: What to expect from {company}",
        f"Technical analysis: {ticker} approaches resistance level",
        f"{company} announces share buyback program"
    ]
    
    # Return mock data
    return {
        "GoogleNews": google_news,
        "YahooFinance": yahoo_headlines,
        "MarketWatch": market_watch
    }


class StockSentimentTool:
    """
    A tool for analyzing sentiment related to specific stocks based on
    news and social media data.
    """
    
    def __init__(self, use_mock: bool = False, cache_dir: str = "./cache"):
        """
        Initialize the Stock Sentiment Analysis Tool.
        
        Args:
            use_mock (bool): Force the use of mock analyzer even if FinBERT is available
            cache_dir (str): Directory to store cached data
        """
        # Initialize sentiment analyzer
        if HAS_FINBERT and not use_mock:
            try:
                logger.info("Initializing FinBERT sentiment analyzer...")
                self.analyzer = FinBERTSentimentAnalyzer()
                logger.info("FinBERT initialized successfully.")
                self.using_finbert = True
            except Exception as e:
                logger.error(f"Error initializing FinBERT: {e}")
                logger.info("Falling back to mock analyzer.")
                self.analyzer = MockSentimentAnalyzer()
                self.using_finbert = False
        else:
            logger.info("Using mock sentiment analyzer.")
            self.analyzer = MockSentimentAnalyzer()
            self.using_finbert = False
            
        # Initialize web scraper if available
        if HAS_WEBSCRAPER and not use_mock:
            try:
                logger.info("Initializing web scraper...")
                self.scraper = WebScraper(cache_dir=cache_dir)
                logger.info("Web scraper initialized successfully.")
                self.using_webscraper = True
            except Exception as e:
                logger.error(f"Error initializing web scraper: {e}")
                logger.info("Falling back to mock data.")
                self.using_webscraper = False
                self.scraper = None
        else:
            logger.info("Using mock data.")
            self.using_webscraper = False
            self.scraper = None
            
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
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
        
        # Get company name
        if self.using_webscraper and self.scraper:
            company_name = self.scraper.get_company_name(ticker).lower()
        else:
            # Fallback mapping
            company_mapping = {
                "AAPL": "apple",
                "MSFT": "microsoft",
                "GOOGL": "google",
                "AMZN": "amazon",
                "META": "meta facebook",
                "TSLA": "tesla",
                "NVDA": "nvidia",
            }
            company_name = company_mapping.get(ticker, ticker).lower()
        
        # Initialize counters
        total_texts = len(texts)
        ticker_mentions = 0
        company_mentions = 0
        
        # Check each text for mentions
        for text in texts:
            text_lower = text.lower()
            
            # Check for ticker mentions
            if ticker.lower() in text_lower:
                ticker_mentions += 1
            
            # Check for company name mentions
            if company_name in text_lower:
                company_mentions += 1
        
        # Calculate relevance score
        relevance_score = (ticker_mentions + 0.5 * company_mentions) / total_texts
        
        # Ensure the score is between 0 and 1
        relevance_score = min(1.0, max(0.0, relevance_score))
        
        return round(relevance_score, 2)
    
    def get_real_data(self, ticker: str) -> Dict[str, List[str]]:
        """
        Get real data for a ticker using the web scraper.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Dict[str, List[str]]: Dictionary with source names as keys and lists of texts as values
        """
        if not self.using_webscraper or not self.scraper:
            logger.warning("Web scraper not available. Using mock data.")
            return generate_mock_data(ticker)
        
        try:
            # Scrape all sources
            logger.info(f"Scraping data for {ticker}...")
            data = self.scraper.scrape_all_sources(ticker)
            
            if not data:
                logger.warning(f"No data found for {ticker}. Using mock data.")
                return generate_mock_data(ticker)
            
            logger.info(f"Found data from {len(data)} sources for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error scraping data for {ticker}: {e}")
            logger.info("Falling back to mock data.")
            return generate_mock_data(ticker)
    
    def analyze_sentiment(self, ticker: str) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Analyze sentiment for a given stock ticker using real data.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict: Dictionary containing sentiment analysis results
        """
        # Get real data for the ticker
        data = self.get_real_data(ticker)
        
        # Flatten all texts for overall analysis
        all_texts = []
        for source_texts in data.values():
            all_texts.extend(source_texts)
        
        if not all_texts:
            logger.warning(f"No texts found for {ticker}")
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "relevance": 0.0,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source_breakdown": [],
                "model": "finbert" if self.using_finbert else "mock",
                "data_source": "real" if self.using_webscraper else "mock"
            }
        
        # Calculate relevance score
        relevance = self.calculate_relevance(all_texts, ticker)
        
        # Analyze sentiment for all texts
        logger.info(f"Analyzing sentiment for {len(all_texts)} texts...")
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
        for source, texts in data.items():
            if not texts:
                continue
                
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
            "source_breakdown": source_results,
            "model": "finbert" if self.using_finbert else "mock",
            "data_source": "real" if self.using_webscraper else "mock"
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze stock sentiment using real-time data.')
    parser.add_argument('ticker', type=str, help='Stock ticker to analyze', nargs='?', default='AAPL')
    parser.add_argument('--mock', '-m', action='store_true', help='Use mock analyzer')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed results')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    args = parser.parse_args()
    
    # Create the sentiment analysis tool
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    sentiment_tool = StockSentimentTool(use_mock=args.mock, cache_dir=cache_dir)
    
    # Analyze sentiment for the ticker
    print(f"\nAnalyzing sentiment for {args.ticker}...\n")
    
    # Get sentiment results
    if args.detailed:
        detailed_results = sentiment_tool.analyze_sentiment(args.ticker)
        print(json.dumps(detailed_results, indent=2))
        result_to_save = detailed_results
    else:
        results_json = sentiment_tool.get_sentiment_json(args.ticker)
        print(results_json)
        result_to_save = json.loads(results_json)
        
        # Get detailed results for display
        detailed_results = sentiment_tool.analyze_sentiment(args.ticker)
        
        # Print source breakdown
        print("\nSource Breakdown:")
        for source in detailed_results["source_breakdown"]:
            print(f"- {source['source']}: {source['sentiment']} (Relevance: {source['relevance']})")
        
        # Print model and data source information
        print(f"\nModel used: {detailed_results['model']}")
        print(f"Data source: {detailed_results['data_source']}")
        
        # Show a sample of the texts
        if detailed_results["data_source"] == "real":
            print("\nSample Texts (from real data):")
        else:
            print("\nSample Texts (from mock data):")
            
        data = sentiment_tool.get_real_data(args.ticker)
            
        for source, texts in data.items():
            if not texts:
                continue
                
            print(f"\n{source}:")
            for i, text in enumerate(texts[:3], 1):  # Show only first 3 items
                print(f"  {i}. {text}")
            if len(texts) > 3:
                print(f"  ... and {len(texts) - 3} more")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result_to_save, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()