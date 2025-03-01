#!/usr/bin/env python
# coding: utf-8

"""
Test script for the WebScraper component of the Stock Sentiment Analysis Tool.
"""

import os
import sys
import argparse
import json
from utils.webscraper import WebScraper

def main():
    """
    Main function to test the web scraper.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the stock sentiment analysis web scraper.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker or company name to analyze")
    parser.add_argument("--source", "-s", choices=["all", "yahoo", "wsj", "marketwatch", "reuters", "seekingalpha"], 
                        default="all", help="Source to scrape (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output file to save results (JSON format)")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    args = parser.parse_args()
    
    # Create cache directory if it doesn't exist
    if args.cache_dir and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    # Initialize the scraper
    scraper = WebScraper(cache_dir=args.cache_dir)
    
    # Display information
    print(f"Testing web scraper for '{args.ticker}'")
    print(f"Source: {args.source}")
    print(f"Cache directory: {args.cache_dir}")
    print()
    
    # Scrape the specified source
    if args.source == "all":
        results = scraper.scrape_all_sources(args.ticker)
    elif args.source == "yahoo":
        results = {"YahooFinance": scraper.scrape_yahoo_finance(args.ticker)}
    elif args.source == "wsj":
        results = {"WSJ": scraper.scrape_wsj(args.ticker)}
    elif args.source == "marketwatch":
        results = {"MarketWatch": scraper.scrape_marketwatch(args.ticker)}
    elif args.source == "reuters":
        results = {"Reuters": scraper.scrape_reuters(args.ticker)}
    elif args.source == "seekingalpha":
        results = {"Seeking Alpha": scraper.scrape_seeking_alpha(args.ticker)}
    
    # Display results
    total_texts = sum(len(texts) for texts in results.values())
    print(f"Found {total_texts} texts from {len(results)} sources")
    
    for source, texts in results.items():
        print(f"\n{source} ({len(texts)} items):")
        if args.verbose:
            # Show all items in verbose mode
            for i, text in enumerate(texts, 1):
                print(f"  {i}. {text}")
        else:
            # Show only the first 5 items
            for i, text in enumerate(texts[:5], 1):
                print(f"  {i}. {text}")
            if len(texts) > 5:
                print(f"  ... and {len(texts) - 5} more")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()