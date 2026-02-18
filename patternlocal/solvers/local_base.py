"""
Base class for local solvers to eliminate duplicate code.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import ComputationalError, ValidationError
from ..utils.distance import calculate_distances
from ..utils.kernels import KernelRegistry
from ..utils.projection import project_point_onto_hyperplane
from .base import BaseSolver


class LocalSolverBase(BaseSolver):
    """Base class for solvers that use local data analysis.

    This class provides common functionality for solvers that:
    1. Project points onto hyperplanes (optional)
    2. Find k-nearest neighbors
    3. Calculate kernel weights based on distances
    4. Extract local data with weights
    """

    # Set of known parameters for local solvers
    KNOWN_PARAMS = {
        "k_ratio",
        "bandwidth",
        "kernel",
        "distance_metric",
        "precomputed_distances",
        "use_projection",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize LocalSolverBase.

        Common parameters:
            - k_ratio: Ratio of samples to use for local estimation (default: 0.1)
            - bandwidth: Kernel bandwidth (default: None, auto-estimate)
            - kernel: Kernel function name (default: 'gaussian')
            - distance_metric: Distance metric (default: 'euclidean')
            - precomputed_distances: Precomputed distances array (default: None)
            - use_projection: Whether to project point onto hyperplane (default: False)
        """
        super().__init__(params)

        # Check for unknown parameters
        unknown_params = set(self.params.keys()) - self.KNOWN_PARAMS
        if unknown_params:
            raise ValidationError(
                f"Unknown parameters for {self.__class__.__name__}: {unknown_params}"
            )

        self.k_ratio = self.params.get("k_ratio", 1.0)
        self.bandwidth = self.params.get("bandwidth", None)
        kernel_name = self.params.get("kernel", "gaussian")
        if not KernelRegistry.is_registered(kernel_name):
            raise ValidationError(
                f"Unknown kernel '{kernel_name}'. \
                    Available kernels: {KernelRegistry.list_available()}"
            )
        self.kernel_function = KernelRegistry.create(kernel_name)
        self.distance_metric = self.params.get("distance_metric", "euclidean")
        self.precomputed_distances = self.params.get("precomputed_distances", None)
        self.use_projection = self.params.get("use_projection", False)

        # Validate parameters
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate local solver parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate k_ratio
        if isinstance(self.k_ratio, float):
            if not 0 < self.k_ratio <= 1:
                raise ValidationError("k_ratio must be between 0 and 1 when float")
        elif isinstance(self.k_ratio, int):
            if self.k_ratio <= 0:
                raise ValidationError("k_ratio must be positive when integer")
        else:
            raise ValidationError(
                "k_ratio must be float between 0-1 or positive integer"
            )

        # Validate bandwidth
        if self.bandwidth is not None:
            if not isinstance(self.bandwidth, (int, float)):
                raise ValidationError("bandwidth must be numeric")
            if self.bandwidth <= 0:
                raise ValidationError("bandwidth must be positive")

        # Validate distance_metric
        valid_metrics = [
            "euclidean",
            "manhattan",
            "cosine",
            "chebyshev",
            "minkowski",
            "dtw",
        ]
        if self.distance_metric not in valid_metrics:
            raise ValidationError(f"distance_metric must be one of {valid_metrics}")

        # Validate precomputed_distances
        if self.precomputed_distances is not None:
            if not isinstance(self.precomputed_distances, np.ndarray):
                raise ValidationError("precomputed_distances must be a numpy array")
            if self.precomputed_distances.ndim != 1:
                raise ValidationError("precomputed_distances must be 1-dimensional")
            if np.any(self.precomputed_distances < 0):
                raise ValidationError("precomputed_distances must be non-negative")
            if np.any(np.isnan(self.precomputed_distances)):
                raise ValidationError("precomputed_distances contains NaN values")
            if np.any(np.isinf(self.precomputed_distances)):
                raise ValidationError("precomputed_distances contains infinite values")

        # Validate use_projection
        if not isinstance(self.use_projection, bool):
            raise ValidationError("use_projection must be boolean")

    def _get_analysis_point(
        self, lime_weights: np.ndarray, lime_intercept: float, instance: np.ndarray
    ) -> np.ndarray:
        """Get the point for local analysis (projected or original).

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained

        Returns:
            Point to use for local analysis
        """
        if self.use_projection:
            return project_point_onto_hyperplane(lime_weights, lime_intercept, instance)
        else:
            return instance

    def _get_local_data_and_weights(self, X_train: np.ndarray, point: np.ndarray):
        """Get local data and corresponding sample weights.

        Args:
            X_train: Training data
            point: Point around which to get local data

        Returns:
            Tuple of (local_data, sample_weights)
        """
        n_total_samples, d = X_train.shape

        # Handle k parameter
        k = self._get_k_neighbors(n_total_samples)

        # Use precomputed distances if available, otherwise compute them
        if self.precomputed_distances is not None:
            # Validate precomputed distances size matches training data
            if len(self.precomputed_distances) != n_total_samples:
                raise ValidationError(
                    f"precomputed_distances length ({len(self.precomputed_distances)}) "
                    f"must match X_train number of samples ({n_total_samples})"
                )
            distances = self.precomputed_distances.copy()
        else:
            # Compute distances using specified metric
            distances = calculate_distances(X_train, point, method=self.distance_metric)

        # Get k nearest neighbors
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:k]

        # Determine bandwidth
        bandwidth = self._get_bandwidth(distances, k_indices)

        # Compute weights
        distances_k = distances[k_indices]
        weights = self.kernel_function(distances_k, bandwidth)

        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ComputationalError(
                "Sum of weights is zero. Consider increasing k or adjusting bandwidth."
            )
        weights /= weight_sum

        # Get local data
        X_local = X_train[k_indices]

        return X_local, weights

    def _get_k_neighbors(self, n_total_samples: int) -> int:
        """Get number of neighbors based on k_ratio parameter.

        Args:
            n_total_samples: Total number of samples available

        Returns:
            Number of neighbors to use
        """
        if isinstance(self.k_ratio, float):
            if not 0 < self.k_ratio <= 1:
                raise ValueError("If k_ratio is a float, it must be between 0 and 1.")
            k = int(self.k_ratio * n_total_samples)
        elif isinstance(self.k_ratio, int):
            if self.k_ratio <= 0 or self.k_ratio > n_total_samples:
                raise ValueError(
                    "k_ratio must be a positive integer less than \
                        or equal to the number of samples."
                )
            k = self.k_ratio
        else:
            raise ValueError(
                "k_ratio must be a float between 0 and 1, or a positive integer."
            )

        return max(1, k)  # Ensure at least 1 neighbor

    def _get_bandwidth(self, distances: np.ndarray, k_indices: np.ndarray) -> float:
        """Get bandwidth for kernel weighting.

        Args:
            distances: All distances from query point
            k_indices: Indices of k nearest neighbors

        Returns:
            Bandwidth for kernel weighting
        """
        if self.bandwidth is None:
            # Auto-estimate bandwidth as median distance of k neighbors
            h = np.median(distances[k_indices])
            if h == 0:
                h = np.finfo(float).eps  # Smallest positive float
        else:
            h = self.bandwidth

        return h
