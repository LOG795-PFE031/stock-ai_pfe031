from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    components: Dict[str, bool]

class MetaInfo(BaseModel):
    """Metadata about the data service."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    version: Optional[str] = None
    message: Optional[str] = None
    documentation: Optional[str] = None
    endpoints: Optional[List[str]] = None

class StockPrice(BaseModel):
    """Schema for a stock price data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None

class StockInfo(BaseModel):
    """Schema for stock information."""
    symbol: str
    name: str
    current_price: Optional[float] = None
    change_percent: Optional[float] = None

class StockDataResponse(BaseModel):
    """Response schema for stock data."""
    symbol: str
    stock_info: StockInfo
    prices: List[StockPrice]
    total_records: int
    meta: MetaInfo

class CurrentPriceResponse(BaseModel):
    """Response schema for current price data."""
    symbol: str
    current_price: float
    change_percent: Optional[float] = None
    timestamp: str
    meta: MetaInfo

class CleanupResponse(BaseModel):
    """Response schema for cleanup operation."""
    message: str
    deleted_records: int
    symbols_affected: List[str]
    meta: MetaInfo 
