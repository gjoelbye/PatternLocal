"""
PatternLocal Hyperparameter Optimization Module

This module provides hyperparameter optimization capabilities for PatternLocal explainers.
It supports various optimization backends and provides search space definitions for
different simplification methods and solvers.

Examples:
    >>> from patternlocal.optimization import OptimizedPatternLocalExplainer
    >>>
    >>> # Basic optimization
    >>> explainer = OptimizedPatternLocalExplainer()
    >>> best_params = explainer.optimize_parameters(
    ...     X_val=X_val,
    ...     masks_val=masks_val,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     model=model,
    ...     metric_fn=metric_function
    ... )
    >>>
    >>> # Use optimized explainer
    >>> explainer.set_best_params(best_params)
    >>> explanation = explainer.explain_instance(instance, model.predict_proba, X_train)
"""

from .backends import GridSearchBackend, HyperoptBackend, OptimizationBackend
from .explainer import OptimizedPatternLocalExplainer
from .metrics import ExplanationMetrics
from .search_spaces import SearchSpaceFactory, get_default_search_space
from .utils import load_optimization_results, save_optimization_results

__all__ = [
    "OptimizedPatternLocalExplainer",
    "SearchSpaceFactory",
    "get_default_search_space",
    "OptimizationBackend",
    "HyperoptBackend",
    "GridSearchBackend",
    "save_optimization_results",
    "load_optimization_results",
    "ExplanationMetrics",
]
