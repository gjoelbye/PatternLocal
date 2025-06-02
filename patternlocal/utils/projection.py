"""
Projection utilities for geometric operations.
"""

from typing import Optional

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


def project_point_onto_line(
    line_point: np.ndarray, line_direction: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Project a point onto a line in n-dimensional space.

    Args:
        line_point: A point on the line, shape (n_features,)
        line_direction: Direction vector of the line, shape (n_features,)
        point: Point to project, shape (n_features,)

    Returns:
        Projected point on the line, shape (n_features,)

    Raises:
        ComputationalError: If line direction is zero or computation fails
    """
    try:
        direction = line_direction.copy()
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            raise ComputationalError("Line direction vector cannot be zero")

        # Normalize direction vector
        direction = direction / direction_norm

        # Vector from line_point to the point to be projected
        point_to_line = point - line_point

        # Project onto the line direction
        projection_length = np.dot(point_to_line, direction)
        projected_point = line_point + projection_length * direction

        return projected_point

    except Exception as e:
        raise ComputationalError(f"Failed to project point onto line: {e}")


def project_points_onto_subspace(
    basis_vectors: np.ndarray, points: np.ndarray, origin: Optional[np.ndarray] = None
) -> np.ndarray:
    """Project points onto a subspace defined by basis vectors.

    Args:
        basis_vectors: Orthonormal basis vectors of the subspace,
                      shape (n_basis, n_features)
        points: Points to project, shape (n_points, n_features)
        origin: Origin of the subspace, shape (n_features,).
                If None, use zero vector.

    Returns:
        Projected points, shape (n_points, n_features)

    Raises:
        ComputationalError: If basis vectors are not orthonormal or computation fails
    """
    try:
        if origin is None:
            origin = np.zeros(basis_vectors.shape[1])

        # Check if basis vectors are orthonormal
        gram_matrix = np.dot(basis_vectors, basis_vectors.T)
        identity = np.eye(basis_vectors.shape[0])

        if not np.allclose(gram_matrix, identity, atol=1e-6):
            # If not orthonormal, orthonormalize using QR decomposition
            Q, _ = np.linalg.qr(basis_vectors.T)
            basis_vectors = Q.T[: basis_vectors.shape[0]]

        # Translate points to subspace origin
        translated_points = points - origin

        # Project onto subspace
        # For each point, project onto each basis vector and sum
        projections = np.zeros_like(translated_points)
        for basis_vector in basis_vectors:
            projection_coeffs = np.dot(translated_points, basis_vector)
            projections += np.outer(projection_coeffs, basis_vector)

        # Translate back
        projected_points = projections + origin

        return projected_points

    except Exception as e:
        raise ComputationalError(f"Failed to project points onto subspace: {e}")


def orthogonal_complement_projection(
    subspace_basis: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Project points onto the orthogonal complement of a subspace.

    Args:
        subspace_basis: Basis vectors of the subspace, shape (n_basis, n_features)
        points: Points to project, shape (n_points, n_features)

    Returns:
        Points projected onto orthogonal complement, shape (n_points, n_features)
    """
    # Project onto subspace
    subspace_projection = project_points_onto_subspace(subspace_basis, points)

    # Orthogonal complement projection = original - subspace projection
    orthogonal_projection = points - subspace_projection

    return orthogonal_projection


def gram_schmidt_orthogonalize(
    vectors: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Orthogonalize a set of vectors using Gram-Schmidt process.

    Args:
        vectors: Input vectors, shape (n_vectors, n_features)
        normalize: Whether to normalize the orthogonal vectors

    Returns:
        Orthogonalized vectors, shape (n_vectors, n_features)

    Raises:
        ComputationalError: If vectors are linearly dependent
    """
    try:
        n_vectors, n_features = vectors.shape
        orthogonal_vectors = np.zeros_like(vectors)

        for i in range(n_vectors):
            # Start with the current vector
            vector = vectors[i].copy()

            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                projection = (
                    np.dot(vector, orthogonal_vectors[j]) * orthogonal_vectors[j]
                )
                vector = vector - projection

            # Check for linear dependence
            vector_norm = np.linalg.norm(vector)
            if vector_norm < 1e-10:
                raise ComputationalError(
                    f"Vector {i} is linearly dependent on previous vectors"
                )

            # Normalize if requested
            if normalize:
                vector = vector / vector_norm

            orthogonal_vectors[i] = vector

        return orthogonal_vectors

    except Exception as e:
        raise ComputationalError(f"Failed to orthogonalize vectors: {e}")


def distance_point_to_hyperplane(
    normal_vector: np.ndarray, intercept: float, point: np.ndarray
) -> float:
    """Calculate the distance from a point to a hyperplane.

    Args:
        normal_vector: Normal vector of the hyperplane, shape (n_features,)
        intercept: Intercept term of the hyperplane
        point: Point, shape (n_features,)

    Returns:
        Distance from point to hyperplane
    """
    normal_norm = np.linalg.norm(normal_vector)
    if normal_norm == 0:
        raise ComputationalError("Normal vector cannot be zero")

    # Distance = |normal · point + intercept| / ||normal||
    distance = abs(np.dot(normal_vector, point) + intercept) / normal_norm

    return distance


def distance_point_to_line(
    line_point: np.ndarray, line_direction: np.ndarray, point: np.ndarray
) -> float:
    """Calculate the distance from a point to a line.

    Args:
        line_point: A point on the line, shape (n_features,)
        line_direction: Direction vector of the line, shape (n_features,)
        point: Point, shape (n_features,)

    Returns:
        Distance from point to line
    """
    direction_norm = np.linalg.norm(line_direction)
    if direction_norm == 0:
        raise ComputationalError("Line direction vector cannot be zero")

    # Vector from line point to the query point
    point_to_line = point - line_point

    # Project onto line direction
    projection_length = np.dot(point_to_line, line_direction) / direction_norm
    projection = projection_length * (line_direction / direction_norm)

    # Distance is the norm of the perpendicular component
    perpendicular = point_to_line - projection
    distance = np.linalg.norm(perpendicular)

    return distance


def reflect_point_across_hyperplane(
    normal_vector: np.ndarray, intercept: float, point: np.ndarray
) -> np.ndarray:
    """Reflect a point across a hyperplane.

    Args:
        normal_vector: Normal vector of the hyperplane, shape (n_features,)
        intercept: Intercept term of the hyperplane
        point: Point to reflect, shape (n_features,)

    Returns:
        Reflected point, shape (n_features,)
    """
    # Project point onto hyperplane
    projected_point = project_point_onto_hyperplane(normal_vector, intercept, point)

    # Reflection = point + 2 * (projection - point) = 2 * projection - point
    reflected_point = 2 * projected_point - point

    return reflected_point
