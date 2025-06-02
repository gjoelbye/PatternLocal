"""
Kernel functions for weighting in pattern computation.
"""

from typing import Callable

import numpy as np


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth (standard deviation)

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def epanechnikov_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Epanechnikov kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    weights = np.maximum(0, 1 - normalized_distances**2)
    return 0.75 * weights


def uniform_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Uniform (rectangular) kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    return (distances <= bandwidth).astype(float)


def triangular_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Triangular kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    return np.maximum(0, 1 - normalized_distances)


def biweight_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Biweight (quartic) kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    inside_support = normalized_distances <= 1
    weights = np.zeros_like(distances)
    weights[inside_support] = (15 / 16) * (
        1 - normalized_distances[inside_support] ** 2
    ) ** 2
    return weights


def tricube_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Tricube kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    inside_support = normalized_distances <= 1
    weights = np.zeros_like(distances)
    weights[inside_support] = (70 / 81) * (
        1 - normalized_distances[inside_support] ** 3
    ) ** 3
    return weights


def cosine_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Cosine kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    inside_support = normalized_distances <= 1
    weights = np.zeros_like(distances)
    weights[inside_support] = (np.pi / 4) * np.cos(
        np.pi / 2 * normalized_distances[inside_support]
    )
    return weights


def logistic_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Logistic kernel function.

    Args:
        distances: Array of distances
        bandwidth: Kernel bandwidth

    Returns:
        Kernel weights
    """
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

    normalized_distances = distances / bandwidth
    return 1 / (np.exp(normalized_distances) + 2 + np.exp(-normalized_distances))


# Registry of available kernels
KERNEL_REGISTRY = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "uniform": uniform_kernel,
    "rectangular": uniform_kernel,  # Alias
    "triangular": triangular_kernel,
    "biweight": biweight_kernel,
    "quartic": biweight_kernel,  # Alias
    "tricube": tricube_kernel,
    "cosine": cosine_kernel,
    "logistic": logistic_kernel,
}


def get_kernel_function(kernel_name: str) -> Callable[[np.ndarray, float], np.ndarray]:
    """Get kernel function by name.

    Args:
        kernel_name: Name of the kernel function

    Returns:
        Kernel function

    Raises:
        ValueError: If kernel name is not recognized
    """
    if kernel_name not in KERNEL_REGISTRY:
        available = list(KERNEL_REGISTRY.keys())
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {available}")

    return KERNEL_REGISTRY[kernel_name]


# def adaptive_bandwidth(distances: np.ndarray,
#                       method: str = 'median',
#                       factor: float = 1.0) -> float:
#     """Estimate adaptive bandwidth from distances.

#     Args:
#         distances: Array of distances
#         method: Method for bandwidth estimation ('median', 'mean', 'std', 'silverman')
#         factor: Scaling factor for the bandwidth

#     Returns:
#         Estimated bandwidth
#     """
#     if len(distances) == 0:
#         return 1.0

#     if method == 'median':
#         bandwidth = np.median(distances)
#     elif method == 'mean':
#         bandwidth = np.mean(distances)
#     elif method == 'std':
#         bandwidth = np.std(distances)
#     elif method == 'silverman':
#         # Silverman's rule of thumb for Gaussian kernels
#         n = len(distances)
#         bandwidth = 1.06 * np.std(distances) * (n ** (-1/5))
#     else:
#         raise ValueError(f"Unknown bandwidth method: {method}")

#     # Ensure positive bandwidth
#     bandwidth = max(bandwidth, np.finfo(float).eps)

#     return factor * bandwidth


# def kernel_density_estimate(query_points: np.ndarray,
#                            data_points: np.ndarray,
#                            kernel_name: str = 'gaussian',
#                            bandwidth: Union[float, str] = 'adaptive') -> np.ndarray:
#     """Estimate kernel density at query points.

#     Args:
#         query_points: Points where to estimate density, shape (n_query, n_features)
#         data_points: Training data points, shape (n_data, n_features)
#         kernel_name: Name of kernel function to use
#         bandwidth: Bandwidth value or 'adaptive' for automatic estimation

#     Returns:
#         Density estimates at query points, shape (n_query,)
#     """
#     kernel_func = get_kernel_function(kernel_name)
#     n_data = data_points.shape[0]
#     densities = np.zeros(query_points.shape[0])

#     for i, query_point in enumerate(query_points):
#         # Calculate distances from query point to all data points
#         distances = np.linalg.norm(data_points - query_point, axis=1)

#         # Estimate bandwidth if needed
#         if isinstance(bandwidth, str) and bandwidth == 'adaptive':
#             h = adaptive_bandwidth(distances, method='median')
#         else:
#             h = bandwidth

#         # Calculate kernel weights and sum for density estimate
#         weights = kernel_func(distances, h)
#         densities[i] = np.sum(weights) / (n_data * h)

#     return densities
