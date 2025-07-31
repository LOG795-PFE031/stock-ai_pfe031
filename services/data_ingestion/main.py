from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from contextlib import asynccontextmanager
import asyncio
import time
from .routes import router
from .data_service import DataService
from core.progress import create_spinner, print_status, print_error
from core.logging import logger
from core.monitor_utils import (
    monitor_cpu_usage,
    monitor_memory_usage,
)
from core.prometheus_metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_errors_total,
)
logger = logger["data"]

# Service instance
data_service = DataService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        spinner = create_spinner("Initializing DataService...")
        spinner.start()
        await data_service.initialize()
        spinner.stop()
        print_status("Success", "DataService initialized successfully", "success")
        yield
    except Exception as e:
        spinner.stop()
        print_error(e)
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        try:
            await data_service.cleanup()
            logger.info("DataService cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="Data Service API",
    description="Microservice for financial data retrieval, storage and management.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Data", "description": "Endpoints for retrieving stock data and prices"},
        {"name": "System", "description": "System health and status endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["System"])
async def root():
    return RedirectResponse(url="/docs")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    start_time = time.time()

    # Start the tasks monitoring saturation (memory and cpu)
    memory_monitoring_task = asyncio.create_task(monitor_memory_usage(method, endpoint))
    cpu_monitoring_task = asyncio.create_task(monitor_cpu_usage(method, endpoint))

    try:
        response: Response = await call_next(request)
    except Exception as e:
        http_errors_total.labels(method=method, endpoint=endpoint).inc()
        raise e

    duration = time.time() - start_time

    # Stop saturation monitoring
    memory_monitoring_task.cancel()
    cpu_monitoring_task.cancel()

    # Record metrics
    http_requests_total.labels(method=method, endpoint=endpoint).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
        duration
    )

    if response.status_code >= 500:
        http_errors_total.labels(method=method, endpoint=endpoint).inc()

    return response

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

app.include_router(router, prefix="/data")  # Add /data prefix to all routes 
