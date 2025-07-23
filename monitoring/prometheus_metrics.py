"""
Monitoring Metrics for StockAI Using Prometheus Client

This module defines Prometheus metrics to monitor various components of StockAI.
"""

from prometheus_client import Counter, Histogram, Gauge

# ===========================================
# Data Monitoring
# ===========================================

external_requests_total = Counter(
    name="external_requests_total",
    documentation="Total number of HTTP requests made by this app to external sites",
    labelnames=["site", "result"],
)

# ===========================================
# Prediction Monitoring
# ===========================================

prediction_confidence = Gauge(
    name="prediction_confidence",
    documentation="Confidence associated to a prediction",
    labelnames=["model_type", "symbol"],
)


# ===========================================
# Evaluation Monitoring
# ===========================================

evaluation_mae = Gauge(
    name="evaluation_mae",
    documentation="Mean Absolute Error of the model",
    labelnames=["model_type", "symbol"],
)

evaluation_mse = Gauge(
    name="model_mse",
    documentation="Mean Squared Error of the model",
    labelnames=["model_type", "symbol"],
)

evaluation_rmse = Gauge(
    name="model_rmse",
    documentation="Root Mean Squared Error of the model",
    labelnames=["model_type", "symbol"],
)

evaluation_r2 = Gauge(
    name="model_r2",
    documentation="R-squared score of the model",
    labelnames=["model_type", "symbol"],
)


# ===========================================
# Sentiment Analysis Monitoring
# ===========================================

sentiment_analysis_time_seconds = Histogram(
    name="sentiment_analysis_time_seconds",
    documentation="Time spent doing sentiment analysis in secs",
    labelnames=["number_articles"],
    buckets=[0.5, 1, 5, 10, 30],
)

# ===========================================
# The Four Golden Signals
# ===========================================

# Traffic (number of requests per endpoint)
http_requests_total = Counter(
    "http_requests_total", "Total number of HTTP requests", ["method", "endpoint"]
)

# Latency (duration per endpoint)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "endpoint"],
)

# Errors
http_errors_total = Counter(
    "http_errors_total", "Total number of HTTP 5xx errors", ["method", "endpoint"]
)

# Saturation (CPU)
cpu_saturation_percentage = Gauge(
    "cpu_saturation_percentage",
    "CPU percentage usage relative to total available CPU capacity.",
    ["method", "endpoint"],
)

# Saturation (RAM)
memory_saturation_mb_usage = Gauge(
    "memory_saturation_usage",
    "Memory percentage usage relative to total available memory in (GB).",
    ["method", "endpoint"],
)
