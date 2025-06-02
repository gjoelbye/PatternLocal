"""
Pattern solvers for PatternLocal.

This module provides various solvers for computing pattern explanations
from LIME weights and local data.
"""

from .base import BaseSolver
from .global_covariance import GlobalCovarianceSolver
from .lasso import LassoSolver
from .local_covariance import LocalCovarianceSolver
from .no_solver import NoSolver
from .registry import SolverRegistry
from .ridge import RidgeSolver

__all__ = [
    "BaseSolver",
    "SolverRegistry",
    "NoSolver",
    "GlobalCovarianceSolver",
    "LocalCovarianceSolver",
    "LassoSolver",
    "RidgeSolver",
]
