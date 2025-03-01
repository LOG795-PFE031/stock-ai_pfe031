#!/usr/bin/env python
# coding: utf-8

"""
Test script for the Stock Sentiment Analysis Tool using mock data.
"""

import os
import sys
import json
import argparse

# Disable FinBERT import
sys.modules['utils.sentiment_analyzer'] = None

# Use mock webscraper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.webscraper_mock import WebScraper

# Mock the webscraper import in stock_sentiment_tool
class MockWebScraperModule:
    WebScraper = WebScraper

sys.modules['utils.webscraper'] = MockWebScraperModule

# Mock sentiment analyzer
class MockSentimentAnalyzer:
    """A mock sentiment analyzer that returns simulated sentiment scores."""
    
    def batch_analyze(self, texts):
        """Generate mock sentiment analysis results."""
        import random
        results = []
        
        for text in texts:
            # Simple heuristic for test: detect positive/negative words
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

# Import stock sentiment tool with our mocks in place
import stock_sentiment_tool
# Replace the StockSentimentTool class with a simple version that uses our mocks
stock_sentiment_tool.HAS_FINBERT = False
stock_sentiment_tool.HAS_WEBSCRAPER = True

class SimpleStockSentimentTool:
    """A simplified version of StockSentimentTool that uses mock components."""
    
    def __init__(self):
        """Initialize with mock components."""
        self.analyzer = MockSentimentAnalyzer()
        self.scraper = WebScraper()
        self.using_finbert = False
        self.using_webscraper = True
    
    def get_real_data(self, ticker):
        """Get mock data for a ticker."""
        return self.scraper.scrape_all_sources(ticker)
    
    def get_mock_data(self, ticker):
        """Get mock data for a ticker."""
        return self.scraper.scrape_all_sources(ticker)
    
    def calculate_relevance(self, texts, ticker):
        """Calculate relevance score based on ticker mentions."""
        if not texts:
            return 0.0
            
        ticker = ticker.strip().upper()
        company_name = self.scraper.get_company_name(ticker).lower()
        
        total_texts = len(texts)
        ticker_mentions = 0
        company_mentions = 0
        
        for text in texts:
            text_lower = text.lower()
            if ticker.lower() in text_lower:
                ticker_mentions += 1
            if company_name in text_lower:
                company_mentions += 1
        
        import random
        relevance = (ticker_mentions + 0.5 * company_mentions) / total_texts
        # Add some randomness
        relevance = min(1.0, max(0.0, relevance + random.uniform(-0.1, 0.2)))
        
        return round(relevance, 2)
    
    def analyze_sentiment(self, ticker):
        """Analyze sentiment for a ticker."""
        from datetime import datetime
        
        # Get data for ticker
        data = self.get_real_data(ticker)
        
        # Flatten all texts
        all_texts = []
        for source_texts in data.values():
            all_texts.extend(source_texts)
        
        if not all_texts:
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "relevance": 0.0,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source_breakdown": [],
                "model": "mock",
                "data_source": "mock"
            }
        
        # Calculate relevance
        relevance = self.calculate_relevance(all_texts, ticker)
        
        # Analyze sentiment
        sentiment_results = self.analyzer.batch_analyze(all_texts)
        
        # Calculate overall sentiment
        total_positive = sum(result['scores']['positive'] for result in sentiment_results)
        total_negative = sum(result['scores']['negative'] for result in sentiment_results)
        total_neutral = sum(result['scores']['neutral'] for result in sentiment_results)
        
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
        
        return {
            "ticker": ticker,
            "sentiment": overall_sentiment,
            "relevance": relevance,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source_breakdown": source_results,
            "model": "mock",
            "data_source": "mock"
        }
    
    def get_sentiment_json(self, ticker):
        """Get sentiment analysis results in JSON format."""
        import json
        results = self.analyze_sentiment(ticker)
        
        simple_output = {
            "sentiment": results["sentiment"],
            "relevance": results["relevance"],
            "date": results["date"]
        }
        
        return json.dumps(simple_output, indent=2)


def main():
    """
    Main function to test the stock sentiment analysis tool with mock data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the stock sentiment analysis tool with mock data.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker or company name to analyze")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed results")
    parser.add_argument("--output", "-o", type=str, help="Output file to save results (JSON format)")
    args = parser.parse_args()
    
    # Create the sentiment analysis tool with mock data
    sentiment_tool = SimpleStockSentimentTool()
    
    # Display information
    print(f"\nAnalyzing sentiment for '{args.ticker}' using mock data...\n")
    
    # Get sentiment results
    if args.detailed:
        detailed_results = sentiment_tool.analyze_sentiment(args.ticker)
        print(json.dumps(detailed_results, indent=2))
    else:
        results_json = sentiment_tool.get_sentiment_json(args.ticker)
        print(results_json)
        
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
            if args.detailed:
                json.dump(detailed_results, f, indent=2)
            else:
                json.dump(json.loads(results_json), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()