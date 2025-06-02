"""
Registry for simplification methods.
"""

from ..utils.registry import BaseRegistry
from .base import BaseSimplification

# Create singleton registry instance
_registry = BaseRegistry(BaseSimplification, "simplification")


class SimplificationRegistry:
    """Registry for managing simplification methods."""

    @classmethod
    def register(cls, name: str):
        """Decorator to register a simplification method."""
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, params=None):
        """Create a simplification method instance."""
        return _registry.create(name, params)

    @classmethod
    def list_available(cls):
        """List all available simplification methods."""
        return _registry.list_available()

    @classmethod
    def is_registered(cls, name: str):
        """Check if a method is registered."""
        return _registry.is_registered(name)
