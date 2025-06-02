"""
PatternLocal: A unified explainer for local pattern-based explanations.

This package provides a clean, modular interface for generating local explanations
using pattern-based methods with various simplification and solver options.
Supports both tabular and image data with advanced features like caching,
parallel processing, and fluent interface.

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
    ...              .with_caching(enabled=True)
    ...              .fit(X_train))
    >>>
    >>> # Configuration from file
    >>> explainer = PatternLocalExplainer.from_config('config.yaml')
    >>>
    >>> # Batch processing
    >>> explanations = explainer.explain_batch(
    ...     instances, predict_fn, X_train, n_jobs=4
    ... )
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
from .utils import (
    ParallelProcessor,
    calculate_distances,
    epanechnikov_kernel,
    gaussian_kernel,
    project_point_onto_hyperplane,
    uniform_kernel,
)

# Parallel processing functions
from .utils.parallel import (
    parallel_cross_validation,
    parallel_explain_instances,
    parallel_fit_simplifications,
    parallel_solver_comparison,
)

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
    "gaussian_kernel",
    "epanechnikov_kernel",
    "uniform_kernel",
    "project_point_onto_hyperplane",
    "ParallelProcessor",
    # Parallel functions
    "parallel_explain_instances",
    "parallel_fit_simplifications",
    "parallel_solver_comparison",
    "parallel_cross_validation",
    # Exceptions
    "PatternLocalError",
    "ConfigurationError",
    "ExplanationError",
    "ValidationError",
    "FittingError",
    "ComputationalError",
]


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
