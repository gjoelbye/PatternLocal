"""
Custom exceptions for PatternLocal.
"""


class PatternLocalError(Exception):
    """Base exception for PatternLocal errors."""


class ConfigurationError(PatternLocalError):
    """Raised for configuration-related errors."""


class ExplanationError(PatternLocalError):
    """Raised during explanation generation."""


class ValidationError(PatternLocalError):
    """Raised for input validation errors."""


class FittingError(PatternLocalError):
    """Raised during fitting process."""


class ComputationalError(PatternLocalError):
    """Raised for computational errors in pattern solvers."""
