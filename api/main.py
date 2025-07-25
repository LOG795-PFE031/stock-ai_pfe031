"""
Main application module for Stock AI.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response
import time

from core.prometheus_metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_errors_total,
)
from api.routes import router
from core.logging import logger
from db.init_db import create_database
from services import (
    DataService,
    DataProcessingService,
    DeploymentService,
    EvaluationService,
    RabbitMQService,
    MonitoringService,
)
from services.orchestration import OrchestrationService
from core.monitor_utils import (
    monitor_cpu_usage,
    monitor_memory_usage,
)

# Create necessary directories
os.makedirs("data/news", exist_ok=True)

# Create service instances in dependency order
data_service = DataService()
data_processing_service = DataProcessingService()
deployment_service = DeploymentService()
evaluation_service = EvaluationService()
orchestation_service = OrchestrationService(
    data_service=data_service,
    data_processing_service=data_processing_service,
    deployment_service=deployment_service,
    evaluation_service=evaluation_service,
)
rabbitmq_service = RabbitMQService()
monitoring_service = MonitoringService(
    deployment_service,
    orchestation_service,
    data_service,
    data_processing_service,
    check_interval_seconds=24 * 60 * 60,  # 86400 sec in a day
    data_interval_seconds=7 * 24 * 60 * 60,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    try:
        logger["main"].info("Starting up services...")

        # Create the tables
        create_database()

        # Initialize services in order of dependencies
        await data_service.initialize()
        await data_processing_service.initialize()
        await deployment_service.initialize()
        await evaluation_service.initialize()
        await orchestation_service.initialize()
        await monitoring_service.initialize()

        logger["main"].info("All services initialized successfully")
        yield

    except Exception as e:
        logger["main"].error(f"Error during startup: {str(e)}")
        raise

    finally:
        # Shutdown
        try:
            logger["main"].info("Shutting down services...")

            # Cleanup in reverse order of initialization
            await monitoring_service.cleanup()
            await orchestation_service.cleanup()
            await evaluation_service.cleanup()
            await deployment_service.cleanup()
            await data_processing_service.cleanup()
            await data_service.cleanup()
            rabbitmq_service.close()  # Close RabbitMQ connection

            logger["main"].info("All services cleaned up successfully")

        except Exception as e:
            logger["main"].error(f"Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="Stock AI API",
    description="""
    API for stock price prediction and analysis, providing comprehensive financial data analysis and ML-powered predictions.
    
    ## Features
    - Real-time stock data retrieval and analysis
    - News sentiment analysis with FinBERT
    - ML-powered price predictions
    - Model training and management
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "System health and status endpoints"},
        {
            "name": "Data Services",
            "description": "Endpoints for retrieving and updating stock and news data",
        },
        {"name": "Model Management", "description": "Endpoints for managing ML models"},
        {
            "name": "Prediction Services",
            "description": "Endpoints for stock price predictions",
        },
        {
            "name": "Training Services",
            "description": "Endpoints for model training and status monitoring",
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
app.include_router(router, prefix="/api")  # Add /api prefix to all routes


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


# Prometheus metrics
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
