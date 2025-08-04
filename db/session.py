"""
Sets up the SQLAlchemy engine and session for PostgreSQL database interaction.
Event-loop aware to handle Prefect tasks running in different event loops.
"""

import asyncio
import weakref
from typing import Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine, AsyncSession

from core.config import config

# Define the URL for the database connection
DATABASE_URL = config.postgres.URL

# ✅ STORE ENGINES PER EVENT LOOP
_engines: Dict[Any, AsyncEngine] = {}
_sessionmakers: Dict[Any, async_sessionmaker] = {}

def _get_event_loop_id():
    """Get a unique identifier for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        return id(loop)
    except RuntimeError:
        # No event loop running, use a default key
        return "no_loop"

def _create_engine() -> AsyncEngine:
    """Create a new async engine with optimal settings."""
    return create_async_engine(
        DATABASE_URL,
        echo=False,                    # ✅ Disable SQL logging for performance
        pool_size=50,                  # ✅ Increased for better concurrency
        max_overflow=20,               # ✅ More overflow connections
        pool_pre_ping=False,           # ✅ Disable ping for speed
        pool_reset_on_return="rollback", # ✅ Faster than commit
        pool_recycle=3600,             # recycle connections every hour
        pool_timeout=10,               # ✅ Faster timeout
        # ✅ Performance optimizations
        connect_args={
            "server_settings": {
                "jit": "off",                    # Disable JIT for consistent performance
                "application_name": "stock-ai",  # Identify connection in logs
                "synchronous_commit": "off",     # ✅ Disable fsync for speed
                "lock_timeout": "1000",          # ✅ 1 second lock timeout
                "statement_timeout": "5000",     # ✅ 5 second query timeout
                "commit_siblings": "5",          # ✅ Allow concurrent commits
                "default_transaction_isolation": "read committed",  # ✅ Default isolation
            },
            "command_timeout": 30,               # Query timeout
        }
    )

def get_engine() -> AsyncEngine:
    """Get or create an engine for the current event loop."""
    loop_id = _get_event_loop_id()
    
    if loop_id not in _engines:
        _engines[loop_id] = _create_engine()
    
    return _engines[loop_id]

def get_async_sessionmaker() -> async_sessionmaker:
    """Get or create a sessionmaker for the current event loop."""
    loop_id = _get_event_loop_id()
    
    if loop_id not in _sessionmakers:
        engine = get_engine()
        _sessionmakers[loop_id] = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,        # ✅ Keep objects fresh after commit
            autoflush=False,               # ✅ Manual flush control for performance
            autocommit=False,              # ✅ Manual transaction control
        )
    
    return _sessionmakers[loop_id]

# ✅ CREATE DYNAMIC ASYNCSESSIONLOCAL FUNCTION  
def AsyncSessionLocal():
    """Get a session from the sessionmaker for the current event loop."""
    sessionmaker = get_async_sessionmaker()
    return sessionmaker()

# ✅ SIMPLE DEPENDENCY FUNCTION  
async def get_session():
    """Get a database session for the current event loop."""
    sessionmaker = get_async_sessionmaker()
    async with sessionmaker() as session:
        yield session

# ✅ CLEANUP FUNCTION FOR GRACEFUL SHUTDOWN
async def cleanup_engines():
    """Clean up all engines gracefully."""
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()
    _sessionmakers.clear()
