from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    components: Dict[str, bool]


class MetaInfo(BaseModel):
    """API metadata information."""

    message: str
    version: str
    documentation: str
    endpoints: List[str]


class TrainingTrainersResponse(BaseModel):
    """Trainers getter response schema."""

    status: str
    types: List[str]
    count: int
    timestamp: str


class TrainingResponse(BaseModel):
    """Model training response."""

    status: str
    symbol: str
    model_type: str
    run_id: str
    run_info: Dict[str, Any]
    training_history: Any
    timestamp: str


class TrainingStatusResponse(BaseModel):
    """Training status response."""

    status: str
    symbol: str
    model_type: str
    timestamp: str
    result: Optional[TrainingResponse] = None
    error: Optional[str] = None


class TrainingTask(BaseModel):
    """Individual training task information."""

    symbol: str
    model_type: str
    status: str
    timestamp: str


class TrainingTasksResponse(BaseModel):
    """List of active training tasks."""

    tasks: List[TrainingTask]
    total_tasks: int
    timestamp: str
