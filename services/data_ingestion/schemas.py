from typing import List, Dict, Optional
from pydantic import BaseModel, Field
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


class StockItem(BaseModel):
    """Infos about a stock"""

    symbol: str
    sector: Optional[str] = "Unknown"
    companyName: Optional[str] = ""
    marketCap: Optional[str] = "N/A"
    lastSalePrice: Optional[str] = "0.00"
    netChange: Optional[str] = "0.00"
    percentageChange: Optional[str] = "0.00%"
    deltaIndicator: Optional[str] = ""


class StocksListDataResponse(BaseModel):
    """Stocks data list data response."""

    count: int
    data: List[StockItem]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StockPrice(BaseModel):
    """Schema for a stock price data point."""

    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int
    Dividends: Optional[float] = 0.0
    Stock_splits: Optional[float] = 0.0


class StockInfo(BaseModel):
    """Schema for stock information."""

    symbol: str
    name: str


class StockDataResponse(BaseModel):
    """Response schema for stock data."""

    stock_info: StockInfo
    prices: List[StockPrice]
    total_records: int
    meta: MetaInfo
    timestamp: str


class CleanupResponse(BaseModel):
    """Response schema for cleanup operation."""

    message: str
    deleted_records: int
    symbols_affected: List[str]
    meta: MetaInfo
