"""
Main application module for the orchestration service (Stock AI).
"""

import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request

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
from .orchestration_service import OrchestrationService
from .routes import router

# Create the orchestration service instance
orchestration_service = OrchestrationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    try:
        logger["orchestration"].info("Starting up the orchestration service...")

        await orchestration_service.initialize()

        logger["orchestration"].info("Orchestration service initialized successfully")
        yield

    except Exception as exception:
        logger["orchestration"].error(
            f"Error during orchestration service startup: {str(exception)}"
        )
        raise

    finally:
        # Shutdown
        try:
            logger["orchestration"].info("Shutting down the orchestration service...")

            # Cleanup the service
            await orchestration_service.cleanup()

            logger["orchestration"].info(
                "The orchestration service was cleaned up successfully"
            )

        except Exception as exception:
            logger["orchestration"].error(
                f"Error during the orchestration service shutdown: {str(exception)}"
            )


# Create FastAPI app
app = FastAPI(
    title="Orchestration Service API",
    description="""
    API for orchestrating ML operations or ML pipelines.
    
    ## Features
    - Run training pipelines
    - Run prediction pipelines
    - Run evaluation pipelines
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "System health and status endpoints"},
        {
            "name": "Training",
            "description": "Endpoints for launching and managing training pipelines",
        },
        {
            "name": "Predictions",
            "description": "Endpoints for running and retrieving model predictions",
        },
    ],
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add root route handler
@app.get("/", tags=["System"])
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# Include routers
app.include_router(router, prefix="/orchestration")  # Add /api prefix to all routes


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


# Prometheus metrics
@app.get("/metrics")
def metrics():
    """Prometheus metrics exposer"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
