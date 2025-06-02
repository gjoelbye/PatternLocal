"""
Distance calculation utilities.
"""

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

DistanceMetric = Literal["euclidean", "manhattan", "cosine", "chebyshev", "minkowski"]


def calculate_distances(
    X: np.ndarray, point: np.ndarray, method: DistanceMetric = "euclidean", **kwargs
) -> np.ndarray:
    """Calculate distances between data points and a query point.

    Args:
        X: Data points, shape (n_samples, n_features)
        point: Query point, shape (n_features,)
        method: Distance metric to use
        **kwargs: Additional arguments for distance calculation

    Returns:
        Distances from each point in X to the query point, shape (n_samples,)
    """
    if method == "euclidean":
        return _euclidean_distances(X, point)
    elif method == "manhattan":
        return _manhattan_distances(X, point)
    elif method == "cosine":
        return _cosine_distances(X, point)
    elif method == "chebyshev":
        return _chebyshev_distances(X, point)
    elif method == "minkowski":
        p = kwargs.get("p", 2)
        return _minkowski_distances(X, point, p)
    else:
        raise ValueError(f"Unknown distance method: {method}")


def _euclidean_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances."""
    # Use sklearn's optimized implementation
    point_2d = point.reshape(1, -1)
    distances = euclidean_distances(X, point_2d)
    return distances.flatten()


def _manhattan_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Calculate Manhattan (L1) distances."""
    # Use sklearn's optimized implementation
    point_2d = point.reshape(1, -1)
    distances = manhattan_distances(X, point_2d)
    return distances.flatten()


def _cosine_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Calculate cosine distances."""
    # Cosine distance = 1 - cosine similarity
    point_2d = point.reshape(1, -1)

    # Normalize vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    point_norm = point_2d / np.linalg.norm(point_2d, axis=1, keepdims=True)

    # Handle zero vectors
    X_norm = np.where(np.isnan(X_norm), 0, X_norm)
    point_norm = np.where(np.isnan(point_norm), 0, point_norm)

    # Calculate cosine similarity
    cosine_sim = np.dot(X_norm, point_norm.T).flatten()

    # Convert to distance (ensuring numerical stability)
    cosine_distances = np.clip(1 - cosine_sim, 0, 2)

    return cosine_distances


def _chebyshev_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Calculate Chebyshev (L∞) distances."""
    point_2d = point.reshape(1, -1)
    distances = cdist(X, point_2d, metric="chebyshev")
    return distances.flatten()


def _minkowski_distances(X: np.ndarray, point: np.ndarray, p: float) -> np.ndarray:
    """Calculate Minkowski distances."""
    point_2d = point.reshape(1, -1)
    distances = cdist(X, point_2d, metric="minkowski", p=p)
    return distances.flatten()
