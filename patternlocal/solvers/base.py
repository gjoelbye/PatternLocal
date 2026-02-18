"""
Base class for patternlocal solvers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import ValidationError


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
    ) -> None:
        """Validate common input arguments.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate array types and shapes
        if not isinstance(lime_weights, np.ndarray):
            raise ValidationError("lime_weights must be a numpy array")
        if not isinstance(instance, np.ndarray):
            raise ValidationError("instance must be a numpy array")
        if not isinstance(X_train, np.ndarray):
            raise ValidationError("X_train must be a numpy array")

        # Validate array dimensions
        if lime_weights.ndim != 1:
            raise ValidationError(
                f"lime_weights must be 1-dimensional, got {lime_weights.ndim}"
            )
        if instance.ndim != 1:
            raise ValidationError(
                f"instance must be 1-dimensional, got {instance.ndim}"
            )
        if X_train.ndim != 2:
            raise ValidationError(f"X_train must be 2-dimensional, got {X_train.ndim}")

        # Validate array contents
        for name, arr in [
            ("lime_weights", lime_weights),
            ("instance", instance),
            ("X_train", X_train),
        ]:
            if np.any(np.isnan(arr)):
                raise ValidationError(f"{name} contains NaN values")
            if np.any(np.isinf(arr)):
                raise ValidationError(f"{name} contains infinite values")

        # Validate intercept
        if not np.isscalar(lime_intercept):
            raise ValidationError("lime_intercept must be a scalar")
        if np.isnan(lime_intercept) or np.isinf(lime_intercept):
            raise ValidationError("lime_intercept must be finite")

        # Validate shapes match
        if lime_weights.shape[0] != instance.shape[0]:
            raise ValidationError("lime_weights and instance must have same length")
        if X_train.shape[1] != instance.shape[0]:
            raise ValidationError(
                "X_train feature dimension must match instance length"
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
