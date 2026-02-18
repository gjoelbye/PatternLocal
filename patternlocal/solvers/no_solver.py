"""
No solver - returns LIME weights unchanged.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseSolver
from .registry import SolverRegistry


@SolverRegistry.register("none")
class NoSolver(BaseSolver):
    """No patternlocal computation - returns LIME weights unchanged.

    This solver serves as a baseline that simply returns the LIME weights
    without any patternlocal transformation. Useful for comparison and
    when only LIME explanations are desired.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize NoSolver.

        Args:
            params: Parameters (unused for this solver)
        """
        super().__init__(params)

    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return LIME weights unchanged.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept (unused)
            instance: The instance being explained (unused)
            X_train: Training data (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            LIME weights unchanged
        """
        # Validate inputs
        self._validate_inputs(lime_weights, lime_intercept, instance, X_train)

        # Return LIME weights unchanged
        return lime_weights.copy()

    @property
    def solver_type(self) -> str:
        """Type of solver."""
        return "baseline"
