# Stock News Scraper

A tool that scrapes the internet for news articles related to a given stock ticker. It gathers articles from multiple financial news sources, extracts the full text of each article, and outputs everything in a JSON format.

## Features

- **Input any stock ticker** to fetch relevant news
- **Multi-source scraping** from 8 financial news sites:
  - Yahoo Finance
  - MarketWatch
  - Seeking Alpha
  - Reuters
  - Google News
  - Financial Times
  - CNBC
  - Barron's
- **Full article text extraction** (not just headlines)
- **Paywall handling** with fallback mechanisms
- **Relevance scoring** prioritizes the most relevant articles
- **JSON output** for easy integration with other systems

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python stock_news_scraper.py AAPL
```

This will find news articles for Apple (AAPL) stock and output them as JSON.

### Advanced Options

```bash
# Save output to a file
python stock_news_scraper.py MSFT --output msft_news.json

# Limit to 3 sources
python stock_news_scraper.py TSLA --sources 3

# Pretty-print the JSON output
python stock_news_scraper.py GOOGL --pretty
```

## Output Format

The tool returns results in JSON format:

```json
{
  "ticker": "AAPL",
  "company": "Apple",
  "timestamp": "2025-03-03 14:34:35",
  "articles": [
    {
      "url": "https://finance.example.com/article123",
      "title": "Apple Reports Record Q1 Earnings",
      "source": "Example Finance",
      "text": "Full article text goes here...",
      "published_date": "2025-03-03",
      "relevance_score": 15
    },
    ...
  ],
  "total_articles": 5
}
```

## Implementation Details

The scraper uses a flexible approach to handle different site structures:

1. **Source Selection**: Tries multiple sources and prioritizes the most relevant results
2. **Article Extraction**: Uses various CSS selectors to locate article content
3. **Text Cleaning**: Removes ads, copyright notices, and excess formatting
4. **Error Handling**: Falls back to alternative methods if the primary approach fails
5. **Rate Limiting**: Includes delays between requests to avoid being blocked

## Notes

- Web scraping respects site rate limits
- Some articles may be behind paywalls and only partial content will be extracted
- The tool adapts to different site layouts but may need updates if sites change significantly