"""
This module benchmarks the performance of training and prediction pipelines for
multiple machine learning models (LSTM, Prophet, XGBoost). It measures request
durations and error counts for both sequential and concurrent tests.

Tests/Benchmarks:
- `test_train()`: Runs sequential training tests for each model.
- `test_prediction()`: Runs sequential prediction tests for each model.
- `test_concurrent_training()`: Runs concurrent training requests for all models.
- `test_concurrent_prediction()`: Runs concurrent prediction requests for all models.
- `run_all_benchmarks()`: Runs and prints all benchmark tests.

This module is used to assess the performance of ML model pipelines through benchmarks
and generate timing and error summaries.
"""

import asyncio
from time import perf_counter
import httpx
import numpy as np

# Url to the application
BASE_URL = "http://localhost:8000"

# Symbol used for the tests
BASE_SYMBOL = "AAPL"

# Models types
MODEL_TYPES = ["lstm", "prophet", "xgboost"]

# Number of times to run the test
REPEATS = 3


def get_training_pipeline_URL(symbol, model_type):
    """Get the training pipeline URL"""
    return f"{BASE_URL}/api/train?symbol={symbol}&model_type={model_type}"


def get_prediction_pipeline_URL(symbol, model_type):
    """Get the prediction pipeline URL"""
    return f"{BASE_URL}/api/predict?symbol={symbol}&model_type={model_type}"


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


def print_model_timing_summary(name, durations: dict, total_errors: int):
    """Print average, min, and max durations per model type, along with total errors."""
    print(f"\n=== {name} Summary ===")
    for model_type, times in durations.items():
        if times:
            print(
                f"⏱️ {model_type.upper()} - Avg: {np.mean(times):.2f}s, Min: {min(times):.2f}s, Max: {max(times):.2f}s"
            )
        else:
            print(f"⚠️ {model_type.upper()} - No successful runs")
    print(f"❌ Total Errors: {total_errors}")


def print_concurrent_test_summary(name, duration, errors):
    """Print total duration and error count for a concurrent test run."""
    print(f"\n=== Concurrent {name} Summary ===")
    print(f"⏱️ Total Duration: {duration:.2f} seconds")
    print(f"❌ Total Errors: {errors}")


async def test_train():
    """Run multiple training requests per model, recording durations and errors."""
    async with httpx.AsyncClient(timeout=None) as client:

        errors = 0
        durations = {"lstm": [], "prophet": [], "xgboost": []}

        for i in range(REPEATS):
            print(f"Training Run {i + 1} of {REPEATS}...")
            for model_type in MODEL_TYPES:
                start_time = perf_counter()
                url, response = await fetch_url(
                    client, get_training_pipeline_URL(BASE_SYMBOL, model_type), "POST"
                )
                if response is None or response.status_code >= 400:
                    print(
                        f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                    )
                    errors += 1

                # Duration of one training
                duration = perf_counter() - start_time

                # Add the duration to the correct model type
                durations[model_type].append(duration)

        return durations, errors


async def test_prediction():
    """Run multiple prediction requests per model, recording durations and errors."""
    async with httpx.AsyncClient(timeout=None) as client:

        errors = 0
        durations = {"lstm": [], "prophet": [], "xgboost": []}

        for i in range(REPEATS):
            print(f"Prediction Run {i + 1} of {REPEATS}...")
            for model_type in MODEL_TYPES:
                start_time = perf_counter()
                url, response = await fetch_url(
                    client, get_prediction_pipeline_URL(BASE_SYMBOL, model_type), "POST"
                )
                if response is None or response.status_code >= 400:
                    print(
                        f"❌ Error for {url} - Status: {getattr(response, 'status_code', 'N/A')}"
                    )
                    errors += 1

                # Duration of one training
                duration = perf_counter() - start_time

                # Add the duration to the correct model type
                durations[model_type].append(duration)

        return durations, errors


async def test_concurrent_training():
    """Run conccurent training requests recording duration and errors."""
    async with httpx.AsyncClient(timeout=None) as client:

        errors = 0
        start_time = perf_counter()

        # Execute concurrent training requests for all the models at the same time
        tasks = [
            fetch_url(
                client, get_training_pipeline_URL(BASE_SYMBOL, model_type), "POST"
            )
            for model_type in MODEL_TYPES
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


async def test_concurrent_prediction():
    """Run conccurent prediction requests recording duration and errors."""
    async with httpx.AsyncClient(timeout=None) as client:

        errors = 0
        start_time = perf_counter()

        # Execute concurrent prediction requests for all the models at the same time
        tasks = [
            fetch_url(
                client, get_prediction_pipeline_URL(BASE_SYMBOL, model_type), "POST"
            )
            for model_type in MODEL_TYPES
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

    # Test training benchmarks for all models
    train_durations, train_errors = await test_train()
    print_model_timing_summary("Training", train_durations, train_errors)

    # Test predictions benchmarks for all models
    prediction_durations, prediction_errors = await test_prediction()
    print_model_timing_summary("Prediction", prediction_durations, prediction_errors)

    # Test concurrent training requests
    duration, errors = await test_concurrent_training()
    print_concurrent_test_summary("Training", duration, errors)

    # Test concurrent prediction requests
    duration, errors = await test_concurrent_prediction()
    print_concurrent_test_summary("Prediction", duration, errors)


asyncio.run(run_all_benchmarks())
