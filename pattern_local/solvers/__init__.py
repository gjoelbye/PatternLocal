"""
Pattern solvers for PatternLocal.

This module provides various solvers for computing pattern explanations
from LIME weights and local data.
"""

from .base import BaseSolver
from .registry import SolverRegistry
from .no_solver import NoSolver
from .global_covariance import GlobalCovarianceSolver
from .local_covariance import LocalCovarianceSolver
from .lasso import LassoSolver
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
