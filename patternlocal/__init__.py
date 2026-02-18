"""
PatternLocal

Examples:
    >>> from patternlocal import PatternLocalExplainer
    >>>
    >>> # Basic usage
    >>> explainer = PatternLocalExplainer(
    ...     simplification='lowrank',
    ...     solver='local_covariance'
    ... )
    >>> explainer.fit(X_train)
    >>> explanation = explainer.explain_instance(instance, predict_fn, X_train)
    >>>
    >>> # Fluent interface
    >>> explainer = (PatternLocalExplainer()
    ...              .with_simplification('lowrank', n_components=10)
    ...              .with_solver('local_covariance', k_ratio=0.1)
    ...              .fit(X_train))
    >>>
    >>> # Configuration from file
    >>> explainer = PatternLocalExplainer.from_config('config.yaml')
"""

# Configuration classes
from .config import (
    ExplainerConfig,
    LimeConfig,
    ParameterValidator,
    SimplificationConfig,
    SolverConfig,
)

# Main explainer class
from .core.explainer import PatternLocalExplainer

# Exceptions
from .exceptions import (
    ComputationalError,
    ConfigurationError,
    ExplanationError,
    FittingError,
    PatternLocalError,
    ValidationError,
)

# Simplification methods
from .simplification import (
    LowRankSimplification,
    NoSimplification,
    SuperpixelSimplification,
)

# Base classes
from .simplification.base import BaseSimplification

# Registries
from .simplification.registry import SimplificationRegistry

# Solvers
from .solvers import (
    GlobalCovarianceSolver,
    LassoSolver,
    LocalCovarianceSolver,
    NoSolver,
    RidgeSolver,
)
from .solvers.base import BaseSolver
from .solvers.registry import SolverRegistry

# Utilities
from .utils import calculate_distances, project_point_onto_hyperplane

# Optimization (optional import)  # noqa: F401
try:
    from .optimization import OptimizedPatternLocalExplainer  # noqa: F401

    _OPTIMIZATION_AVAILABLE = True  # noqa: F401
except ImportError:
    _OPTIMIZATION_AVAILABLE = False  # noqa: F401

# Package metadata
__version__ = "2.0.0"
__author__ = "PatternXAI Team"
__description__ = "Advanced pattern-based explanations with modern architecture"

# Main exports
__all__ = [
    # Main explainer
    "PatternLocalExplainer",
    # Configuration
    "ExplainerConfig",
    "LimeConfig",
    "SimplificationConfig",
    "SolverConfig",
    "ParameterValidator",
    # Base classes
    "BaseSimplification",
    "BaseSolver",
    # Simplification methods
    "NoSimplification",
    "LowRankSimplification",
    "SuperpixelSimplification",
    # Solvers
    "NoSolver",
    "GlobalCovarianceSolver",
    "LocalCovarianceSolver",
    "LassoSolver",
    "RidgeSolver",
    # Registries
    "SimplificationRegistry",
    "SolverRegistry",
    # Utilities
    "calculate_distances",
    "project_point_onto_hyperplane",
    # Exceptions
    "PatternLocalError",
    "ConfigurationError",
    "ExplanationError",
    "ValidationError",
    "FittingError",
    "ComputationalError",
]

# Add optimization components if available
if _OPTIMIZATION_AVAILABLE:
    __all__.append("OptimizedPatternLocalExplainer")


def list_simplification_methods():
    """List all available simplification methods.

    Returns:
        List of available simplification method names
    """
    return SimplificationRegistry.list_available()


def list_solvers():
    """List all available solver methods.

    Returns:
        List of available solver method names
    """
    return SolverRegistry.list_available()


def get_package_info():
    """Get information about the package.

    Returns:
        Dictionary with package information
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "simplification_methods": list_simplification_methods(),
        "solver_methods": list_solvers(),
    }


# Auto-register built-in methods on import
def _register_builtin_methods():
    """Register built-in simplification and solver methods."""
    # Import to trigger registration decorators


# Register built-in methods
_register_builtin_methods()
