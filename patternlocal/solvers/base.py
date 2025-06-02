"""
Base class for patternlocal solvers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..config.validation import ParameterValidator


class BaseSolver(ABC):
    """Abstract base class for patternlocal solvers.

    patternlocal solvers take LIME weights and compute patternlocal explanations
     using various methods (local covariance, Lasso, Ridge, etc.).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the patternlocal solver.

        Args:
            params: Patternlocal solver-specific parameters
        """
        self.params = params or {}

    @abstractmethod
    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute patternlocal weights from LIME weights.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data
            **kwargs: Additional patternlocal solver-specific arguments

        Returns:
           patternlocal explanation weights
        """

    def _validate_inputs(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
    ):
        """Validate common input arguments.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data

        Raises:
            ValidationError: If inputs are invalid
        """
        ParameterValidator.validate_solver_inputs(
            lime_weights, lime_intercept, instance, X_train
        )

    @property
    def solver_type(self) -> str:
        """Type of solver."""
        return self.__class__.__name__

    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the solver.

        Returns:
            Dictionary with patternlocal solver information
        """
        return {"type": self.solver_type, "params": self.params.copy()}
