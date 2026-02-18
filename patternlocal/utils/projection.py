"""
Projection utilities for geometric operations.
"""

import numpy as np

from ..exceptions import ComputationalError


def project_point_onto_hyperplane(
    normal_vector: np.ndarray,
    intercept: float,
    point: np.ndarray,
    normalize_normal: bool = True,
) -> np.ndarray:
    """Project a point onto a hyperplane defined by normal vector and intercept.

    The hyperplane is defined as: normal_vector · x + intercept = 0

    Args:
        normal_vector: Normal vector of the hyperplane, shape (n_features,)
        intercept: Intercept term of the hyperplane
        point: Point to project, shape (n_features,)
        normalize_normal: Whether to normalize the normal vector

    Returns:
        Projected point on the hyperplane, shape (n_features,)

    Raises:
        ComputationalError: If normal vector is zero or computation fails
    """
    try:
        normal = normal_vector.copy()

        # Normalize the normal vector if requested
        if normalize_normal:
            normal_norm = np.linalg.norm(normal)
            if normal_norm == 0:
                raise ComputationalError("Normal vector cannot be zero")
            normal = normal / normal_norm
            # Adjust intercept for normalized normal
            intercept = intercept / normal_norm

        # Calculate the distance from point to hyperplane
        # Distance = (normal · point + intercept) / ||normal||
        # Since normal is normalized (||normal|| = 1), distance = normal ·
        # point + intercept
        distance_to_plane = np.dot(normal, point) + intercept

        # Project point onto hyperplane
        # projected_point = point - distance * normal
        projected_point = point - distance_to_plane * normal

        return projected_point

    except Exception as e:
        raise ComputationalError(f"Failed to project point onto hyperplane: {e}")
