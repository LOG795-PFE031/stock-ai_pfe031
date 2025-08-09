"""
This module benchmarks the performance of a simulated user journey on a stock trading platform.

Tests/Benchmarks:
- `test_user_journey()`: Simulates a single user journey, including fetching current stock data,
    historical data, news, and prediction.
- `test_multiple_user_journeys()`: Runs multiple simulated user journeys, printing the average,
    minimum, and maximum durations, and error counts.

The module is designed to assess the responsiveness and error rates of different API endpoints.
"""

import asyncio
from time import perf_counter
import httpx
import numpy as np

# Url to the application
BASE_URL = "http://localhost:8000"

# Represents the poular stocks (usually shown at the Home Menu of the UI)
POPULAR_SYMBOLS = ["AAPL", "GOOG", "AMZN", "IBM", "TSLA"]

# Represents the symbol selected by the user in the UI
BASE_SYMBOL = POPULAR_SYMBOLS[0]

# Dates used for the historical stock data of the BASE_SYMBOL
START_DATE = "2025-07-01"  # Start of July
END_DATE = "2025-07-31"  # End of July

# Model used in this test
MODEL_TYPE = "prophet"

# Number of times to run the test
REPEATS = 5


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


async def test_user_journey(client) -> tuple:
    """Run a single simulated user journey and return duration and error count."""
    try:
        start_time = perf_counter()
        errors = 0

        # Home Page: Current stock data
        tasks = [
            fetch_url(
                client, f"{BASE_URL}/api/data/stock/current?symbol={symbol}", "GET"
            )
            for symbol in POPULAR_SYMBOLS
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

        # Detail View: Historical, News, Prediction
        tasks = [
            fetch_url(
                client,
                f"{BASE_URL}/api/data/stock/historical?symbol={BASE_SYMBOL}&start_date={START_DATE}&end_date={END_DATE}",
                "GET",
            ),
            fetch_url(client, f"{BASE_URL}/api/data/news?symbol={BASE_SYMBOL}", "GET"),
            fetch_url(
                client,
                f"{BASE_URL}/api/predict?symbol={BASE_SYMBOL}&model_type={MODEL_TYPE}",
                "POST",
            ),
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

        # Get duration of the user journey
        duration = perf_counter() - start_time

        return duration, errors

    except Exception as e:
        print(f"Error in test_user_journey : {str(e)}")
        raise e


async def test_multiple_user_journeys():
    """Run the user journey test multiple times and print average duration and errors."""
    try:
        durations = []
        error_counts = []

        try:
            async with httpx.AsyncClient(timeout=None) as client:

                # Train the base model
                print(
                    f"Training the model {MODEL_TYPE}_{BASE_SYMBOL} to ensure its existence..."
                )
                await fetch_url(
                    client,
                    f"{BASE_URL}/api/train?symbol={BASE_SYMBOL}&model_type={MODEL_TYPE}",
                    "POST",
                )

                for i in range(REPEATS):
                    print(f"Run {i + 1} of {REPEATS}")
                    duration, errors = await test_user_journey(client)
                    durations.append(duration)
                    error_counts.append(errors)
                    print(f"Duration: {duration:.2f}s, ❌ Errors: {errors}\n")
        except Exception as e:
            print(f"Error in test_user_journey : {str(e)}")
            durations.append(None)
            error_counts.append(None)

        # Show summary stats
        print("=== Test Summary ===")
        print(f"Runs: {REPEATS}")
        print(f"Avg Duration: {np.mean(durations):.2f}s")
        print(f"Min Duration: {min(durations):.2f}s")
        print(f"Max Duration: {max(durations):.2f}s")
        print(f"Avg Errors: {np.mean(error_counts):.1f}")
        print(f"Total Errors: {sum(error_counts)}")
    except Exception as e:
        print(f"Error in test_multiple_user_journeys : {str(e)}")
        raise e


# Execute the test
asyncio.run(test_multiple_user_journeys())
