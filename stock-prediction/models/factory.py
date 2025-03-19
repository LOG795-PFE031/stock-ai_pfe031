"""
Model factory for creating and managing stock prediction models.
"""
from typing import Dict, Any, Optional, Type
from .base import BaseModel
from .lstm import LSTMModel
from .prophet import ProphetModel

class ModelFactory:
    """Factory class for creating stock prediction models"""
    
    # Registry of available model types
    _models: Dict[str, Type[BaseModel]] = {
        'lstm': LSTMModel,
        'prophet': ProphetModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Any) -> Optional[BaseModel]:
        """
        Create a model instance based on the specified type
        
        Args:
            model_type: Type of model to create ('lstm' or 'prophet')
            config: Configuration object containing model parameters
            
        Returns:
            Instance of the requested model type
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type.lower() not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types are: {', '.join(cls._models.keys())}")
        
        model_class = cls._models[model_type.lower()]
        return model_class(config)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type
        
        Args:
            name: Name of the model type
            model_class: Model class to register
            
        Raises:
            ValueError: If model_class is not a subclass of BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must be a subclass of BaseModel")
        
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """
        Get dictionary of available model types
        
        Returns:
            Dictionary mapping model names to their classes
        """
        return cls._models.copy() 