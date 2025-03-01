# Stock Sentiment Analysis with Real-Time Web Scraping

This module enhances the stock sentiment analysis tool by adding real-time web scraping capabilities to gather financial news and social media data related to specific stocks. It replaces mock data with actual content from authoritative sources and analyzes sentiment using the FinBERT model.

## Features

- **Real-time data collection** from financial news sources including:
  - Yahoo Finance
  - Market Watch
  - Wall Street Journal
  - Reuters
  - Seeking Alpha
- **Smart caching system** to minimize redundant requests
- **Relevance scoring** based on ticker and company name mentions
- **Sentiment analysis** using pre-trained FinBERT model
- **Detailed source breakdown** of sentiment by data source
- **Fallback mechanisms** when web scraping fails

## Installation

Install the required dependencies:

```bash
pip install -r requirements-webscraper.txt
```

## Usage

### Basic Usage

```bash
python stock_sentiment_tool.py AAPL
```

This will analyze the sentiment for Apple (AAPL) stock based on real-time data and display the results.

### Options

- **Ticker/Company Name**: Provide a stock ticker or company name as the first argument
- **-mock**: Use mock data instead of real-time data (useful for testing)
- **-detailed**: Show detailed results in JSON format

```bash
# Analyze sentiment for Microsoft with detailed output
python stock_sentiment_tool.py MSFT -detailed

# Test with mock data for Tesla
python stock_sentiment_tool.py TSLA -mock
```

### For Testing

For testing without making actual web requests:

```bash
python test_mock_sentiment.py AAPL
```

## Output Format

The tool returns sentiment analysis results in JSON format:

```json
{
  "sentiment": "positive",
  "relevance": 0.85,
  "date": "2025-02-27"
}
```

- **sentiment**: Overall sentiment (positive, neutral, negative)
- **relevance**: Relevance score (0-1) indicating how relevant the collected data is to the specified stock
- **date**: Date of analysis

## Architecture

The implementation consists of two main components:

1. **WebScraper Module** (`utils/webscraper.py`):
   - Handles real-time data collection from multiple sources
   - Implements caching to avoid redundant requests
   - Provides fallback mechanisms
   
2. **Stock Sentiment Tool** (`stock_sentiment_tool.py`):
   - Orchestrates data collection and sentiment analysis
   - Integrates with the FinBERT sentiment analyzer
   - Calculates relevance scores and aggregates results

## Notes

- Web scraping respects rate limits of the target sites
- The tool uses a cache to minimize redundant requests (cache expires after 1 hour by default)
- If web scraping fails for any reason, the tool falls back to mock data
- The sentiment analysis uses FinBERT, a model fine-tuned for financial text