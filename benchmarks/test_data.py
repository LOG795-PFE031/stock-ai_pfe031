"""
This module benchmarks various stock data API endpoints (news, current, and historical)
by sending sequential and concurrent requests for multiple stock symbols.

Benchmarked Endpoints:
- /api/data/news
- /api/data/stock/current
- /api/data/stock/historical

Tests Included:
- Sequential and concurrent requests for current and historical stock data
- Sequential requests for news data
"""

import asyncio
from time import perf_counter
import httpx
import numpy as np


# Url to the application
BASE_URL = "http://localhost:8000"

# Number of times to run the test
REPEATS = 3

# Symbols to test
SYMBOLS = [
    # Tech stocks
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "META",
    "NFLX",
    "NVDA",
    "TSLA",
    "INTC",
    "AMD",
    "IBM",
    "ORCL",
    "CRM",
    "ADBE",
    "CSCO",
    # Financial stocks
    "JPM",
    "BAC",
    "WFC",
    "C",
    "GS",
    "V",
    "MA",
    "AXP",
    # Consumer goods & retail
    "PG",
    "KO",
    "PEP",
    "WMT",
    "TGT",
    "COST",
    "MCD",
    # Healthcare
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "ABBV",
    # Other sectors
    "DIS",
    "VZ",
    "T",
    "XOM",
    "CVX",
]

# Dates used for the historical stock data
START_DATE = "2025-07-01"  # Start of July
END_DATE = "2025-07-31"  # End of July


def print_test_summary(name, durations, errors):
    """Print total duration and error count for test run."""
    print(f"\n=== {name} Test Summary ===")
    if durations:
        print(f"⏱️ Avg Duration: {np.mean(durations):.2f}s")
        print(f"⏱️ Min Duration: {min(durations):.2f}s")
        print(f"⏱️ Max Duration: {max(durations):.2f}s")
    else:
        print("⚠️ No durations recorded.")
    print(f"❌ Total Errors: {errors}")


def print_concurrent_test_summary(name, duration, errors):
    """Print total duration and error count for a concurrent test run."""
    print(f"\n=== Concurrent {name} Summary ===")
    print(f"⏱️ Total Duration: {duration:.2f} seconds")
    print(f"❌ Total Errors: {errors}")


def get_news_data_URL(symbol):
    """Get the news data URL"""
    return f"{BASE_URL}/api/data/news?symbol={symbol}"


def get_current_stock_data_URL(symbol):
    """Get the current stock data URL"""
    return f"{BASE_URL}/api/data/stock/current?symbol={symbol}"


def get_historical_stock_data_URL(symbol):
    """Get the historical stock data URL"""
    return f"{BASE_URL}/api/data/stock/historical?symbol={symbol}&start_date={START_DATE}&end_date={END_DATE}"


async def fetch_url(client, url, method):
    """Send an HTTP request (GET or POST) to the given URL and return the URL and response."""
    try:
        if method == "GET":
            response = await client.get(url)
        else:
            response = await client.post(url)
        return url, response
    except httpx.RequestError as e:
        print(f"❌ Network error for {url}: {e}")
        return url, None


async def test_news():
    """Run multiple news data requests, recording durations and error count."""
    durations = []
    errors = 0

    async with httpx.AsyncClient(timeout=None) as client:
        for symbol in SYMBOLS:
            start_time = perf_counter()

            # Await the async fetch_url call
            url, response = await fetch_url(client, get_news_data_URL(symbol), "GET")

            if response is None or response.status_code >= 400:
                print(
                    f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                )
                errors += 1

            duration = perf_counter() - start_time
            durations.append(duration)

        return durations, errors


async def test_current_stock_data():
    """Send sequential requests for current stock data and return durations and errors."""
    durations = []
    errors = 0

    async with httpx.AsyncClient(timeout=None) as client:
        for symbol in SYMBOLS:
            start_time = perf_counter()

            # Await the async fetch_url call
            url, response = await fetch_url(
                client, get_current_stock_data_URL(symbol), "GET"
            )

            if response is None or response.status_code >= 400:
                print(
                    f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                )
                errors += 1

            duration = perf_counter() - start_time
            durations.append(duration)

        return durations, errors


async def test_historical_stock_data():
    """Send sequential requests for historical stock data and return durations and errors."""
    durations = []
    errors = 0

    async with httpx.AsyncClient(timeout=None) as client:
        for symbol in SYMBOLS:
            start_time = perf_counter()

            # Await the async fetch_url call
            url, response = await fetch_url(
                client, get_historical_stock_data_URL(symbol), "GET"
            )

            if response is None or response.status_code >= 400:
                print(
                    f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                )
                errors += 1

            duration = perf_counter() - start_time
            durations.append(duration)

        return durations, errors


async def test_concurrent_current_stock_data():
    """Send concurrent requests for current stock data and return total duration and errors."""
    start_time = perf_counter()
    errors = 0

    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            fetch_url(client, get_current_stock_data_URL(symbol), "GET")
            for symbol in SYMBOLS
        ]

        # Wait for all the tasks
        responses = await asyncio.gather(*tasks)
        for url, response in responses:
            # Check the errors
            if response is None or response.status_code >= 400:
                print(
                    f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                )
                errors += 1

        # Total Duration
        duration = perf_counter() - start_time

        return duration, errors


async def test_concurrent_historical_stock_data():
    """Send concurrent requests for historical stock data and return total duration and errors."""
    start_time = perf_counter()
    errors = 0

    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            fetch_url(client, get_historical_stock_data_URL(symbol), "GET")
            for symbol in SYMBOLS
        ]

        # Wait for all the tasks
        responses = await asyncio.gather(*tasks)
        for url, response in responses:
            # Check the errors
            if response is None or response.status_code >= 400:
                print(
                    f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                )
                errors += 1

        # Total Duration
        duration = perf_counter() - start_time

        return duration, errors


async def run_all_benchmarks():
    """Run all the benchmark tests defined in this module."""

    # Test news benchmarks for multiple symbols
    news_durations, news_errors = await test_news()
    print_test_summary("News", news_durations, news_errors)

    # Test current stock data benchmarks for multiple symbols
    current_data_durations, current_data_errors = await test_current_stock_data()
    print_test_summary(
        "Current Stock Data", current_data_durations, current_data_errors
    )

    # Test historical stock data benchmarks for multiple symbols
    historical_data_durations, historical_data_errors = (
        await test_historical_stock_data()
    )
    print_test_summary(
        "Historical Stock Data", historical_data_durations, historical_data_errors
    )

    # Test concurrent current stock data requests
    current_concurrent_data_duration, current_concurrent_data_errors = (
        await test_concurrent_current_stock_data()
    )
    print_concurrent_test_summary(
        "Current Data", current_concurrent_data_duration, current_concurrent_data_errors
    )

    # Test concurrent historical stock data requests
    historical_concurrent_data_duration, historical_concurrent_data_errors = (
        await test_concurrent_current_stock_data()
    )
    print_concurrent_test_summary(
        "Historical Data",
        historical_concurrent_data_duration,
        historical_concurrent_data_errors,
    )


asyncio.run(run_all_benchmarks())
