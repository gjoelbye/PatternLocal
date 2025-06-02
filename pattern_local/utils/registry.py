"""
Generic registry base class to eliminate duplication.
"""

from typing import Any, Dict, Generic, Optional, Type, TypeVar

from ..exceptions import ConfigurationError

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Generic registry for managing method instances."""

    def __init__(self, base_class: Type[T], registry_name: str):
        """Initialize registry.

        Args:
            base_class: Base class that registered methods must inherit from
            registry_name: Name for error messages
        """
        self._registry: Dict[str, Type[T]] = {}
        self._base_class = base_class
        self._registry_name = registry_name

    def register(self, name: str):
        """Decorator to register a method.

        Args:
            name: Name to register the method under

        Returns:
            Decorator function
        """

        def wrapper(method_cls: Type[T]):
            if not issubclass(method_cls, self._base_class):
                raise ConfigurationError(
                    f"Class {
                        method_cls.__name__} must inherit from {
                        self._base_class.__name__}"
                )
            self._registry[name] = method_cls
            return method_cls

        return wrapper

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> T:
        """Create a method instance.

        Args:
            name: Name of the registered method
            params: Parameters for the method

        Returns:
            Method instance

        Raises:
            ConfigurationError: If method name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ConfigurationError(
                f"Unknown {
                    self._registry_name} method: {name}. Available: {available}"
            )

        return self._registry[name](params)

    def list_available(self) -> list:
        """List all available methods."""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a method is registered."""
        return name in self._registry
