#!/usr/bin/env python
# coding: utf-8

"""
Stock Sentiment Analysis Tool

This module analyzes sentiment for a given stock ticker or name by processing
text data from financial news and social media sources using the FinBERT model.
"""

import json
import random
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
import re

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
    from utils.webscraper import WebScraper
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
        results = []
        
        for text in texts:
            # Simple heuristic: check for positive/negative keywords
            positive_words = ["bullish", "rose", "strong", "beats", "up", "buy", "moon", "crushing", 
                             "positive", "growth", "gain", "profit", "success", "improve", "boost"]
            negative_words = ["bearish", "scrutiny", "concerns", "worried", "down", "sell", "lost", "disappointed",
                             "negative", "decline", "loss", "risk", "trouble", "drop", "fall"]
            
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
        else:
            logger.info("Using mock data.")
            self.using_webscraper = False
            
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize source list
        self.sources = ["WallStreet Journal", "YahooFinance", "MarketWatch", "Reuters", "Seeking Alpha"]
        
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
        
        marketwatch_templates = [
            f"{company} stock rises after analyst upgrade.",
            f"Opinion: Why {company} is a buy right now.",
            f"{company} reports record revenue, but margins disappoint.",
            f"10 reasons why {company} could be the next big tech winner.",
            f"Analysts cut price targets for {company} after earnings miss."
        ]
        
        reuters_templates = [
            f"{company} considering acquisitions to boost growth.",
            f"{ticker} shares fall after rival announces new competitive product.",
            f"{company} beats Wall Street forecasts, shares jump.",
            f"Exclusive: {company} in talks to acquire startup for $2 billion.",
            f"{company} CEO says company is on track for strong growth in coming year."
        ]
        
        seeking_alpha_templates = [
            f"{company}: Buy The Dip Before Earnings",
            f"Why {company} Is Primed For A Breakout",
            f"{ticker}: Valuation Concerns Amid Slowing Growth",
            f"{company}'s Competitive Position Remains Strong Despite Challenges",
            f"Q1 Earnings Preview: What To Expect From {company}"
        ]
        
        # Randomly select 2-3 texts from each template list (to speed up analysis)
        mock_data = {
            "WallStreet Journal": random.sample(wsj_templates, random.randint(2, min(3, len(wsj_templates)))),
            "YahooFinance": random.sample(yahoo_templates, random.randint(2, min(3, len(yahoo_templates)))),
            "MarketWatch": random.sample(marketwatch_templates, random.randint(2, min(3, len(marketwatch_templates)))),
            "Reuters": random.sample(reuters_templates, random.randint(2, min(3, len(reuters_templates)))),
            "Seeking Alpha": random.sample(seeking_alpha_templates, random.randint(2, min(3, len(seeking_alpha_templates))))
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
        
        # Try to get company name
        try:
            from utils.webscraper import WebScraper
            company_name = WebScraper().get_company_name(ticker).lower()
        except:
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
        high_relevance = 0  # Counter for highly relevant texts
        
        # Patterns for ticker mentions
        ticker_patterns = [
            re.compile(r'\b' + re.escape(ticker) + r'\b', re.IGNORECASE),  # Exact ticker match
            re.compile(r'\$' + re.escape(ticker) + r'\b', re.IGNORECASE),  # $TICKER format
            re.compile(r'#' + re.escape(ticker) + r'\b', re.IGNORECASE)    # #TICKER format
        ]
        
        # Check each text for mentions
        for text in texts:
            text_lower = text.lower()
            text_relevance = 0
            
            # Check for ticker mentions
            for pattern in ticker_patterns:
                if pattern.search(text):
                    ticker_mentions += 1
                    text_relevance += 2  # Higher weight for ticker mentions
                    break
            
            # Check for company name mentions
            if company_name in text_lower:
                company_mentions += 1
                text_relevance += 1
            
            # If the text has a high relevance score, increment the high_relevance counter
            if text_relevance >= 2:
                high_relevance += 1
        
        # Calculate relevance score using a combination of factors
        mention_ratio = (ticker_mentions + 0.5 * company_mentions) / (total_texts + 0.1)
        high_relevance_ratio = high_relevance / (total_texts + 0.1)
        
        # Combine the factors with appropriate weights
        relevance_score = 0.6 * mention_ratio + 0.4 * high_relevance_ratio
        
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
        if not self.using_webscraper:
            logger.warning("Web scraper not available. Using mock data.")
            return self.get_mock_data(ticker)
        
        try:
            # Scrape all sources
            logger.info(f"Scraping data for {ticker}...")
            data = self.scraper.scrape_all_sources(ticker)
            
            if not data:
                logger.warning(f"No data found for {ticker}. Using mock data.")
                return self.get_mock_data(ticker)
            
            logger.info(f"Found data from {len(data)} sources for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error scraping data for {ticker}: {e}")
            logger.info("Falling back to mock data.")
            return self.get_mock_data(ticker)
    
    def analyze_sentiment(self, ticker: str) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Analyze sentiment for a given stock ticker using real or mock data.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict: Dictionary containing sentiment analysis results
        """
        # Get real or mock data for the ticker
        if self.using_webscraper:
            data = self.get_real_data(ticker)
        else:
            data = self.get_mock_data(ticker)
        
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
    # Parse command line arguments
    ticker = "AAPL"  # Default ticker
    use_mock = False
    show_detailed = False
    
    # Simple command line argument parsing
    args = sys.argv[1:]
    if args:
        for i, arg in enumerate(args):
            if arg.startswith("-"):
                if arg == "-mock":
                    use_mock = True
                elif arg == "-detailed":
                    show_detailed = True
            elif i == 0 and not arg.startswith("-"):
                ticker = arg
    
    # Create the sentiment analysis tool
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    sentiment_tool = StockSentimentTool(use_mock=use_mock, cache_dir=cache_dir)
    
    # Analyze sentiment for the ticker
    print(f"\nAnalyzing sentiment for {ticker}...\n")
    
    # Get sentiment results
    if show_detailed:
        detailed_results = sentiment_tool.analyze_sentiment(ticker)
        print(json.dumps(detailed_results, indent=2))
    else:
        results_json = sentiment_tool.get_sentiment_json(ticker)
        print(results_json)
        
        # Get detailed results for display
        detailed_results = sentiment_tool.analyze_sentiment(ticker)
        
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
            
        if sentiment_tool.using_webscraper:
            data = sentiment_tool.get_real_data(ticker)
        else:
            data = sentiment_tool.get_mock_data(ticker)
            
        for source, texts in data.items():
            if not texts:
                continue
                
            print(f"\n{source}:")
            for i, text in enumerate(texts[:3], 1):  # Show only first 3 items
                print(f"  {i}. {text}")
            if len(texts) > 3:
                print(f"  ... and {len(texts) - 3} more")


if __name__ == "__main__":
    main()