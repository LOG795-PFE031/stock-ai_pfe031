from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from contextlib import asynccontextmanager

from .routes import router
from .news_service import NewsService
from core.progress import create_spinner, print_status, print_error
from core.logging import logger
# from core.monitor_utils import (
#     monitor_cpu_usage,
#     monitor_memory_usage,
# )
# from core.prometheus_metrics import (
#     http_requests_total,
#     http_request_duration_seconds,
#     http_errors_total,
# )
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

# @app.middleware("http")
# async def prometheus_middleware(request, call_next):
#     response = await call_next(request)
#     if request.url.path == "/metrics":
#         response.headers["Content-Type"] = CONTENT_TYPE_LATEST
#     return response


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

app.include_router(router) 
