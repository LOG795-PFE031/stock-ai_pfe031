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

predictions_total = Counter(
    name="predictions_total",
    documentation="Total number of predictions",
    labelnames=["model_type", "symbol", "result"],
)

prediction_time_seconds = Histogram(
    name="prediction_time_seconds",
    documentation="Time spent to make a prediction in secs",
    labelnames=["model_type", "symbol"],
    buckets=[0.5, 1, 3, 5, 10],
)

prediction_confidence = Gauge(
    name="prediction_confidence",
    documentation="Confidence associated to a prediction",
    labelnames=["model_type", "symbol"],
)

# ===========================================
# Training Monitoring
# ===========================================

training_total = Counter(
    name="training_total",
    documentation="Total number of trainings",
    labelnames=["model_type", "symbol", "result"],
)

training_time_seconds = Histogram(
    name="training_time_seconds",
    documentation="Time spent training in secs",
    labelnames=["model_type", "symbol"],
    buckets=[5, 10, 30, 60, 150],
)

training_cpu_usage_percent = Gauge(
    name="training_cpu_usage_percent",
    documentation="CPU usage of training process",
    labelnames=["model_type", "symbol"],
)

training_memory_mb_usage_percent = Gauge(
    name="training_memory_mb_usage_percent",
    documentation="Memory usage of training process",
    labelnames=["model_type", "symbol"],
)

# ===========================================
# Preprocessing Monitoring
# ===========================================

preprocessing_time_seconds = Histogram(
    name="preprocessing_time_seconds",
    documentation="Time spent preprocessing in secs",
    labelnames=["model_type", "symbol"],
    buckets=[5, 10, 30, 60, 150],
)

data_points_ingested_total = Counter(
    name="data_points_ingested_total",
    documentation="Total number of data points ingested during preprocessing",
    labelnames=["model_type", "symbol"],
)

# ===========================================
# Evaluation Monitoring
# ===========================================

evaluation_time_seconds = Histogram(
    name="evaluation_time_seconds",
    documentation="Time spent evaluating in secs",
    labelnames=["model_type", "symbol"],
    buckets=[5, 10, 30, 60, 150],
)

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
# Model Saving Monitoring
# ===========================================

model_saving_time_seconds = Histogram(
    name="model_saving_time_seconds",
    documentation="Time spent saving model in secs",
    labelnames=["model_type", "symbol"],
    buckets=[5, 10, 30, 60, 150],
)

# ===========================================
# Sentiment Analysis Monitoring
# ===========================================

sentiment_analysis_time_seconds = Histogram(
    name="sentiment_analysis_time_seconds",
    documentation="Time spent doing sentiment analysis in secs",
    labelnames=["number_articles"],
    buckets=[5, 10, 30, 60, 150],
)

# ===========================================
# The Four Golden Signals
# ===========================================

# Traffic (number of requests per endpoint)
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint"]
)

# Latency (duration per endpoint)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "endpoint"]
)

# Errors
http_errors_total = Counter(
    "http_errors_total",
    "Total number of HTTP 5xx errors",
    ["method", "endpoint"]
)

# Saturation in Training Monitoring above