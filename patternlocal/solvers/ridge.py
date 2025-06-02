"""
Ridge solver for patternlocal computation.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge

from ..exceptions import ComputationalError
from .local_base import LocalSolverBase
from .registry import SolverRegistry


@SolverRegistry.register("ridge")
class RidgeSolver(LocalSolverBase):
    """Ridge patternlocal solver.

    It fits a Ridge model to predict LIME weights from the local data features,
    weighted by proximity to the instance being explained.

    The Ridge coefficients provide the patternlocal weights.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize RidgeSolver.

        Args:
            params: Parameters for Ridge solver
                - alpha: Ridge regularization parameter (default: 1.0)
                - fit_intercept: Whether to fit intercept in Ridge (default: False)
                Plus all LocalSolverBase parameters (k_ratio, bandwidth, etc.)
        """
        super().__init__(params)

        self.alpha = self.params.get("alpha", 1.0)
        self.fit_intercept = self.params.get("fit_intercept", False)

    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute patternlocal weights using local Ridge regression.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data
            **kwargs: Additional arguments (unused)

        Returns:
           patternlocal explanation weights (Ridge coefficients)
        """
        self._validate_inputs(lime_weights, lime_intercept, instance, X_train)

        try:
            # Get the point for local estimation
            point = self._get_analysis_point(lime_weights, lime_intercept, instance)

            # Get local data and weights
            X_local, sample_weights = self._get_local_data_and_weights(X_train, point)

            # Create target: predict LIME score for each local sample
            # Target is the LIME prediction for each local sample
            y_target = X_local @ lime_weights + lime_intercept

            # Fit Ridge model: X_local (features) -> y_target (LIME predictions)
            # We want to find coefficients that map features to LIME
            # predictions
            ridge = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

            # Fit Ridge with sample weights
            ridge.fit(X_local, y_target, sample_weight=sample_weights)

            return ridge.coef_

        except Exception as e:
            raise ComputationalError(f"Error computing Ridge patternlocal: {e}")
