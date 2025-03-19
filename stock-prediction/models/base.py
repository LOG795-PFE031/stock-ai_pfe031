"""
Base model interface for stock prediction models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all stock prediction models"""
    
    def __init__(self, config: Any):
        """
        Initialize the model
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
    
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, 
              train_data: Union[np.ndarray, pd.DataFrame],
              val_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Training data
            val_data: Optional validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, 
                data: Union[np.ndarray, pd.DataFrame],
                **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path from where to load the model
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 test_data: Union[np.ndarray, pd.DataFrame],
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def preprocess_data(self, 
                       data: Union[np.ndarray, pd.DataFrame],
                       **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess input data
        
        Args:
            data: Input data to preprocess
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed data
        """
        return data  # Default implementation returns data as is
    
    def postprocess_predictions(self, 
                              predictions: np.ndarray,
                              **kwargs) -> np.ndarray:
        """
        Postprocess model predictions
        
        Args:
            predictions: Raw model predictions
            **kwargs: Additional postprocessing parameters
            
        Returns:
            Postprocessed predictions
        """
        return predictions  # Default implementation returns predictions as is 