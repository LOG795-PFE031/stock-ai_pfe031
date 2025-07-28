"""
Main application module for Stock AI.
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
    prediction_confidence,
)
from .routes import router
from .deployment_service import DeploymentService

# Create the deployment service instance
deployment_service = DeploymentService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    try:
        logger["deployment"].info("Starting up the deployment service...")

        await deployment_service.initialize()

        logger["main"].info("Deployment service initialized successfully")
        yield

    except Exception as exception:
        logger["main"].error(f"Error during deployment service startup: {str(exception)}")
        raise

    finally:
        # Shutdown
        try:
            logger["main"].info("Shutting down the deployment service...")

            # Cleanup the services
            await deployment_service.cleanup()

            logger["main"].info("The deployment service was cleaned up successfully")

        except Exception as exception:
            logger["main"].error(
                f"Error during the deployment service shutdown: {str(exception)}"
            )
            

app = FastAPI(
    title="Deployment Service API",
    description="""
    API for serving/deploying ML models and predictions.
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "System health and status endpoints"},
        {
            "name": "Deployment Services",
            "description": "Endpoints to list models and get predictions",
        },
    ],
    # debug=True,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
app.include_router(router, prefix="/deployment") 


# Prometheus middleware
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