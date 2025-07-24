"""
Base service class for Stock AI services.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from core.config import config
from core.logging import logger

class BaseService(ABC):
    """Base class for all services in the Stock AI system."""
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return True
    
    def format_error_response(self, error: Exception) -> Dict[str, Any]:
        """Format error response."""
        return {
            "status": "error",
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def format_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format success response."""
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        } 