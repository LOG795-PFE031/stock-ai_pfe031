import asyncio
import psutil

from monitoring.prometheus_metrics import (
    training_cpu_usage_percent,
    training_memory_mb_usage_percent,
)

# Interval of monitoring
MONITOR_INTERVAL_SECONDS = 5


# CPU usage monitor
async def monitor_training_cpu_usage(model_type, symbol):
    """
    Monitor the cpu usage while training a model (Prometheus)
    """
    process = psutil.Process()
    gauge = training_cpu_usage_percent.labels(symbol=symbol, model_type=model_type)

    # Prime the measurement (first call returns 0.0)
    process.cpu_percent(interval=None)

    while True:
        cpu = process.cpu_percent(interval=None)
        gauge.set(cpu)
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)


async def monitor_training_memory_usage(symbol: str, model_type: str):
    """
    Monitor the memory usage while training a model (Prometheus)
    """
    process = psutil.Process()
    gauge = training_memory_mb_usage_percent.labels(
        symbol=symbol, model_type=model_type
    )

    while True:
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024  # Convert bytes to MB
        gauge.set(memory_mb)
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
