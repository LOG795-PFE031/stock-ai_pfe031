from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.core.config import Settings
from src.api.routes import router
from src.core.logging import setup_logging
import uvicorn
import os

def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="News Analyzer Service")
    
    # Setup logging
    setup_logging()
    
    @app.get("/")
    async def root():
        """Redirect to API documentation"""
        return RedirectResponse(url="/docs")
    
    # Include routers
    app.include_router(router, prefix="/api")
    
    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8092))
    host = os.environ.get("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port) 