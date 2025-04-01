"""
Main application module for Stock AI.
"""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from api.routes import router
from core.logging import logger

# Create necessary directories
os.makedirs("data/stock", exist_ok=True)
os.makedirs("data/news", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Initialize services
from services.data_service import DataService
from services.model_service import ModelService
from services.news_service import NewsService
from services.training_service import TrainingService
from services.prediction_service import PredictionService

# Create service instances in dependency order
data_service = DataService()
model_service = ModelService()
news_service = NewsService()
training_service = TrainingService(model_service=model_service, data_service=data_service)
prediction_service = PredictionService(model_service=model_service, data_service=data_service)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    try:
        logger['main'].info("Starting up services...")
        
        # Initialize services in order of dependencies
        await data_service.initialize()  # No dependencies
        await model_service.initialize()  # Depends on data_service
        await news_service.initialize()  # Depends on data_service
        await training_service.initialize()  # Depends on data_service and model_service
        await prediction_service.initialize()  # Depends on data_service and model_service
        
        logger['main'].info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger['main'].error(f"Error during startup: {str(e)}")
        raise
        
    finally:
        # Shutdown
        try:
            logger['main'].info("Shutting down services...")
            
            # Cleanup in reverse order of initialization
            await prediction_service.cleanup()
            await training_service.cleanup()
            await news_service.cleanup()
            await model_service.cleanup()
            await data_service.cleanup()
            
            logger['main'].info("All services cleaned up successfully")
            
        except Exception as e:
            logger['main'].error(f"Error during shutdown: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Stock AI API",
    description="API for stock price prediction and analysis",
    version="1.0.0",
    lifespan=lifespan
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
@app.get("/")
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

# Include routers
app.include_router(router, prefix="/api")  # Add /api prefix to all routes

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 