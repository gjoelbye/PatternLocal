"""
Global covariance solver - baseline pattern method.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import ComputationalError
from .base import BaseSolver
from .registry import SolverRegistry


@SolverRegistry.register("global_covariance")
class GlobalCovarianceSolver(BaseSolver):
    """Global covariance pattern solver.

    This solver computes a global covariance matrix from the training data
    and applies it to the LIME weights. This serves as a baseline approach
    for patternlocal methods.

    patternlocal weights: a = w @ C_global
    where w are LIME weights and C_global is the global covariance matrix.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize GlobalCovarianceSolver.

        Args:
            params: Parameters for global covariance
                - regularization: Regularization parameter for covariance matrix (default: 1e-6)
        """
        super().__init__(params)
        self.regularization = self.params.get("regularization", 1e-6)

    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute pattern weights using global covariance.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept (unused)
            instance: The instance being explained (unused)
            X_train: Training data
            **kwargs: Additional arguments (unused)

        Returns:
           patternlocal explanation weights
        """
        self._validate_inputs(lime_weights, lime_intercept, instance, X_train)

        try:
            # Compute global covariance matrix
            global_cov = np.cov(X_train, rowvar=False)

            # Add regularization to prevent singular matrices
            if self.regularization > 0:
                global_cov += self.regularization * np.eye(global_cov.shape[0])

            # Apply to LIME weights
            patternlocal_weights = lime_weights @ global_cov

            return patternlocal_weights

        except np.linalg.LinAlgError as e:
            raise ComputationalError(
                f"Linear algebra error in global covariance computation: {e}"
            )
        except Exception as e:
            raise ComputationalError(f"Error computing global covariance pattern: {e}")
