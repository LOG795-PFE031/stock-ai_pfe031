from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class ModelVersionInfo(BaseModel):
    version: str
    stage: Optional[str]
    status: Optional[str]
    run_id: Optional[str]
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]


class ModelMlflowInfo(BaseModel):
    name: str
    description: Optional[str]
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]
    tags: Dict[str, str]
    aliases: Dict[str, Any]
    latest_versions: List[ModelVersionInfo]
