"""
Base trainer class for model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union

from core.types import LSTMInput, ProphetInput, XGBoostInput


class BaseTrainer(ABC):
    """Base class for all model trainers."""

    @abstractmethod
    async def train(
        self, data: Union[LSTMInput, ProphetInput, XGBoostInput], **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train the model."""
        pass
