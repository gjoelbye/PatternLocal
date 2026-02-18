"""
Registry for patternlocal solver methods.
"""

from ..utils.registry import BaseRegistry
from .base import BaseSolver

# Create singleton registry instance
_registry = BaseRegistry(BaseSolver, "solver")


class SolverRegistry:
    """Registry for managing solver methods."""

    @classmethod
    def register(cls, name: str):
        """Decorator to register a solver method."""
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, params=None):
        """Create a solver method instance."""
        return _registry.create(name, params)

    @classmethod
    def list_available(cls):
        """List all available solver methods."""
        return _registry.list_available()

    @classmethod
    def is_registered(cls, name: str):
        """Check if a method is registered."""
        return _registry.is_registered(name)
