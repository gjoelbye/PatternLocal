"""
Lasso solver for patternlocal computation.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Lasso

from ..exceptions import ComputationalError, ValidationError
from .local_base import LocalSolverBase
from .registry import SolverRegistry


@SolverRegistry.register("lasso")
class LassoSolver(LocalSolverBase):
    """Lasso patternlocal solver.

    It fits a Lasso model to predict LIME weights from the local data features,
    weighted by proximity to the instance being explained.

    The Lasso coefficients provide the patternlocal weights.
    """

    # Set of known parameters for Lasso solver
    KNOWN_PARAMS = LocalSolverBase.KNOWN_PARAMS | {
        "alpha",
        "max_iter",
        "fit_intercept",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize LassoSolver.

        Args:
            params: Parameters for Lasso solver
                - alpha: Lasso regularization parameter (default: 1.0)
                - max_iter: Maximum iterations for Lasso (default: 1000)
                - fit_intercept: Whether to fit intercept in Lasso (default: False)
                Plus all LocalSolverBase parameters (k_ratio, bandwidth, kernel,
                distance_metric, precomputed_distances, use_projection)
        """
        # Initialize parameters before validation
        params = params or {}
        self.alpha = params.get("alpha", 1.0)
        self.max_iter = params.get("max_iter", 1000)
        self.fit_intercept = params.get("fit_intercept", True)

        # Check for unknown parameters before calling super().__init__
        unknown_params = set(params.keys()) - self.KNOWN_PARAMS
        if unknown_params:
            raise ValidationError(
                f"Unknown parameters for {self.__class__.__name__}: {unknown_params}"
            )

        super().__init__(params)

        # Validate parameters
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate Lasso solver parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        # Call parent validation first
        super()._validate_params()

        # Validate alpha
        if not isinstance(self.alpha, (int, float)):
            raise ValidationError("alpha must be numeric")
        if self.alpha < 0:
            raise ValidationError("alpha must be non-negative")

        # Validate max_iter
        if not isinstance(self.max_iter, int):
            raise ValidationError("max_iter must be integer")
        if self.max_iter <= 0:
            raise ValidationError("max_iter must be positive")

        # Validate fit_intercept
        if not isinstance(self.fit_intercept, bool):
            raise ValidationError("fit_intercept must be boolean")

    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute patternlocal weights using local Lasso regression.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data
            **kwargs: Additional arguments (unused)

        Returns:
           patternlocal explanation weights (Lasso coefficients)
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

            # Fit Lasso model: X_local (features) -> y_target (LIME predictions)
            # We want to find coefficients that map features to LIME
            # predictions
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
            )

            # Fit Lasso with sample weights
            lasso.fit(X_local, y_target, sample_weight=sample_weights)

            return lasso.coef_

        except Exception as e:
            raise ComputationalError(f"Error computing Lasso patternlocal: {e}")
