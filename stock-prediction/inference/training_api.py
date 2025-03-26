"""
API endpoints for model training operations.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os
import glob

from ..training.trainer import ModelTrainer
from ..core.config import Config

# Create FastAPI router
router = APIRouter(prefix="/api/training", tags=["training"])

# Define request/response models
class TrainingRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to train models for")
    model_types: List[str] = Field(..., description="List of model types to train (lstm, prophet)")
    force_retrain: bool = Field(False, description="Whether to force retraining of existing models")

class TrainingResponse(BaseModel):
    status: str = Field(..., description="Training status (success/failed)")
    message: str = Field(..., description="Status message")
    results: Dict[str, Any] = Field(..., description="Training results for each symbol and model type")
    start_time: Optional[str] = Field(None, description="Training start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Training end time (ISO format)")

class TrainingStatus(BaseModel):
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Status message")
    last_training: Optional[Dict[str, Any]] = Field(None, description="Last training information")

@router.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """Train models for specified symbols and types"""
    try:
        # Validate request
        if not request.symbols or not request.model_types:
            raise HTTPException(
                status_code=400,
                detail="Both symbols and model_types are required"
            )
        
        # Initialize trainer
        config = Config()
        trainer = ModelTrainer(config)
        
        # Start training
        start_time = datetime.now()
        results = trainer.train_all_models(
            symbols=request.symbols,
            model_types=request.model_types,
            force_retrain=request.force_retrain
        )
        end_time = datetime.now()
        
        # Prepare response
        return {
            'status': 'success',
            'message': f'Training completed for {len(request.symbols)} symbols',
            'results': results,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
            
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get status of recent training operations"""
    try:
        # Get logs directory from config
        config = Config()
        logs_dir = config.data.LOGS_DIR
        
        # Find most recent training log
        log_files = glob.glob(os.path.join(logs_dir, 'training_*.log'))
        if not log_files:
            return {
                'status': 'no_training_history',
                'message': 'No training history found',
                'last_training': None
            }
        
        latest_log = max(log_files, key=os.path.getctime)
        
        # Read last few lines of log
        with open(latest_log, 'r') as f:
            lines = f.readlines()[-10:]  # Last 10 lines
        
        return {
            'status': 'success',
            'message': 'Training status retrieved successfully',
            'last_training': {
                'log_file': latest_log,
                'last_lines': lines,
                'timestamp': datetime.fromtimestamp(os.path.getctime(latest_log)).isoformat()
            }
        }
            
    except Exception as e:
        logging.error(f"Error getting training status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        ) 