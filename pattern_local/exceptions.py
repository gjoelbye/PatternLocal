"""
Custom exceptions for PatternLocal.
"""


class PatternLocalError(Exception):
    """Base exception for PatternLocal errors."""
    pass


class ConfigurationError(PatternLocalError):
    """Raised for configuration-related errors."""
    pass


class ExplanationError(PatternLocalError):
    """Raised during explanation generation."""
    pass


class ValidationError(PatternLocalError):
    """Raised for input validation errors."""
    pass


class FittingError(PatternLocalError):
    """Raised during fitting process."""
    pass


class ComputationalError(PatternLocalError):
    """Raised for computational errors in pattern solvers."""
    pass 