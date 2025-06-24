from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from training.trainers.base_trainer import BaseTrainer


class TrainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name, **kwargs) -> "BaseTrainer":
        if name not in cls._registry:
            available = cls.list_trainers()
            raise ValueError(
                f"Model '{name}' not found in registry. Available trainers: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_trainers(cls):
        return list(cls._registry.keys())

    @classmethod
    def get_registry(cls):
        """Get the current registry for debugging"""
        return cls._registry.copy()
