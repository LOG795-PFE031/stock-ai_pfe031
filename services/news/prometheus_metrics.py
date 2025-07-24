from prometheus_client import Counter, Histogram

external_requests_total = Counter(
    "external_requests_total",
    "Total number of external requests (e.g., Yahoo Finance)",
    ["site", "result"],
)

sentiment_analysis_time_seconds = Histogram(
    "sentiment_analysis_time_seconds",
    "Time spent analyzing sentiment (seconds)",
    ["number_articles"],
) 
