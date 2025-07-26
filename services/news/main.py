from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from contextlib import asynccontextmanager
import asyncio
import time
from .routes import router
from .news_service import NewsService
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
logger = logger["news"]

# Service instance
news_service = NewsService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        spinner = create_spinner("Initializing NewsService...")
        spinner.start()
        await news_service.initialize()
        spinner.stop()
        print_status("Success", "NewsService initialized successfully", "success")
        yield
    except Exception as e:
        spinner.stop()
        print_error(e)
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        try:
            await news_service.cleanup()
            logger.info("NewsService cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="News Service API",
    description="Microservice for financial news retrieval and sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "News", "description": "Endpoints for retrieving news and sentiment analysis"},
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

app.include_router(router, prefix="/data")  # Add /news prefix to all routes
