import asyncio
import psutil

from monitoring.prometheus_metrics import (
    cpu_saturation_percentage,
    memory_saturation_mb_usage,
)

# Interval of monitoring
MONITOR_INTERVAL_SECONDS = 1


# CPU usage monitor
async def monitor_cpu_usage(method, endpoint):
    """
    Monitors the CPU usage and updates the Prometheus metrics for the specified method and endpoint.

    Args:
        - method (str): The HTTP method (e.g., 'GET', 'POST') to associate with the CPU usage
            metric.
        - endpoint (str): The endpoint URL to associate with the CPU usage metric.
    """
    while True:
        cpu_percentage = psutil.cpu_percent(interval=1)
        cpu_saturation_percentage.labels(method=method, endpoint=endpoint).set(
            cpu_percentage
        )
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)


async def monitor_memory_usage(method, endpoint):
    """
    Monitors memory usage and updates the Prometheus metrics for the specified method and
    endpoint.

    Args:
        - method (str): The HTTP method (e.g., 'GET', 'POST') to associate with the memory usage
            metric.
        - endpoint (str): The endpoint URL or route to associate with the memory usage metric.
    """
    while True:
        mem_info = psutil.virtual_memory()
        mem_mb_usage = round(mem_info.used / (1024**3), 2)
        memory_saturation_mb_usage.labels(method=method, endpoint=endpoint).set(
            mem_mb_usage
        )
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
