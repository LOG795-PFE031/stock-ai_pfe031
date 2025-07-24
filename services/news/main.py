import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from contextlib import asynccontextmanager

from routes import router
from news_service import NewsService
from progress import create_spinner, print_status, print_error
from news_logging import get_logger

logger = get_logger("news_service")

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

app.include_router(router) 
