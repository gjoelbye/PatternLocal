"""
Local covariance solver - main patternlocal method.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import ComputationalError
from .local_base import LocalSolverBase
from .registry import SolverRegistry


@SolverRegistry.register("local_covariance")
class LocalCovarianceSolver(LocalSolverBase):
    """Local covariance patternlocal solver.

    This is the main patternlocal method that estimates a local covariance matrix
    using weighted k-nearest neighbors and applies it to LIME weights.

    patternlocal weights: a = w @ C_local
    where w are LIME weights and C_local is the locally estimated covariance matrix.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize LocalCovarianceSolver.

        Args:
            params: Parameters for local covariance estimation
                - k_ratio: Ratio of samples to use for local estimation (default: 0.1)
                - bandwidth: Kernel bandwidth (default: None, auto-estimate)
                - kernel_function: Kernel function (default: gaussian_kernel)
                - shrinkage_intensity: Shrinkage regularization (default: 0.0)
                - distance_metric: Distance metric (default: 'euclidean')
                - use_projection: Whether to project point onto hyperplane (default: True)
        """
        super().__init__(params)
        self.shrinkage_intensity = self.params.get("shrinkage_intensity", 0.0)

    def solve(
        self,
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute patternlocal weights using local covariance estimation.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data
            **kwargs: Additional arguments (unused)

        Returns:
           patternlocal explanation weights
        """
        self._validate_inputs(lime_weights, lime_intercept, instance, X_train)

        try:
            # Get the point for local estimation
            point = self._get_analysis_point(lime_weights, lime_intercept, instance)

            # Estimate local covariance matrix
            local_cov = self._estimate_local_covariance(X_train, point)

            # Apply to LIME weights
            patternlocal_weights = lime_weights @ local_cov

            return patternlocal_weights

        except Exception as e:
            raise ComputationalError(
                f"Error computing local covariance patternlocal: {e}"
            )

    def _estimate_local_covariance(
        self, X_train: np.ndarray, point: np.ndarray
    ) -> np.ndarray:
        """Estimate local covariance matrix using weighted k-nearest neighbors.

        Args:
            X_train: Training data
            point: Point around which to estimate covariance

        Returns:
            Local covariance matrix
        """
        # Get local data and weights using base class method
        nearest_points, weights = self._get_local_data_and_weights(X_train, point)

        # Compute weighted covariance matrix
        return self._compute_weighted_covariance(nearest_points, weights)

    def _compute_weighted_covariance(
        self, points: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted covariance matrix with optional shrinkage."""
        # Compute weighted mean
        weighted_mean = np.average(points, weights=weights, axis=0)

        # Compute weighted differences
        diffs = points - weighted_mean
        weighted_diffs = weights[:, np.newaxis] * diffs

        # Compute weighted covariance
        cov = weighted_diffs.T @ diffs

        # Apply shrinkage regularization if specified
        if self.shrinkage_intensity > 0:
            if not 0 <= self.shrinkage_intensity <= 1:
                raise ValueError("shrinkage_intensity must be between 0 and 1.")

            # Shrinkage target: diagonal matrix with average variance
            avg_variance = np.mean(np.diag(cov))
            target = avg_variance * np.eye(cov.shape[0])

            cov = (
                1 - self.shrinkage_intensity
            ) * cov + self.shrinkage_intensity * target

        return cov

    @property
    def solver_type(self) -> str:
        """Type of solver."""
        return "patternlocal"
