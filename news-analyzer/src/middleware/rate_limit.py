from fastapi import Request, HTTPException
from datetime import datetime, timedelta
from typing import Dict, List
import time
import logging
from src.core.config import Settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_interval = 60  # Clean up old requests every 60 seconds
        self.last_cleanup = time.time()

    async def __call__(self, request: Request, call_next):
        # Clean up old requests if needed
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )

        # Add current request timestamp
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)

        # Process the request
        response = await call_next(request)
        return response

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if the client has exceeded the rate limit."""
        current_time = time.time()
        window_start = current_time - self.settings.RATE_LIMIT_PERIOD

        # Get requests within the time window
        if client_ip in self.requests:
            recent_requests = [t for t in self.requests[client_ip] if t > window_start]
            if len(recent_requests) >= self.settings.RATE_LIMIT:
                return False

        return True

    def _cleanup_old_requests(self, current_time: float):
        """Remove requests older than the rate limit window."""
        window_start = current_time - self.settings.RATE_LIMIT_PERIOD
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if t > window_start
            ]
            if not self.requests[client_ip]:
                del self.requests[client_ip] 