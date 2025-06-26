"""
Base trainer class for model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from core.types import FormattedInput


class BaseTrainer(ABC):
    """Base class for all model trainers."""

    @abstractmethod
    async def train(self, data: FormattedInput, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Train the model."""
        pass
