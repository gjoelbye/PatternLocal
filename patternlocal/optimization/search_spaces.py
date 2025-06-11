"""
Search space definitions for PatternLocal hyperparameter optimization.

This module defines the search spaces for different simplification methods
and solvers that can be optimized.
"""

from typing import Any, Dict, List

import numpy as np

try:
    from hyperopt import hp

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


class SearchSpaceFactory:
    """Factory for creating hyperparameter search spaces."""

    @staticmethod
    def get_simplification_space(
        method: str, backend: str = "hyperopt"
    ) -> Dict[str, Any]:
        """Get search space for a simplification method.

        Args:
            method: Simplification method name
            backend: Optimization backend ('hyperopt', 'grid')

        Returns:
            Search space dictionary
        """
        if backend not in ["hyperopt", "grid"]:
            raise ValueError(f"Unsupported backend: {backend}")

        if method == "lowrank":
            return SearchSpaceFactory._get_lowrank_space(backend)
        elif method == "superpixel":
            return SearchSpaceFactory._get_superpixel_space(backend)
        elif method == "none":
            return {}
        else:
            raise ValueError(f"Unknown simplification method: {method}")

    @staticmethod
    def get_solver_space(method: str, backend: str = "hyperopt") -> Dict[str, Any]:
        """Get search space for a solver method.

        Args:
            method: Solver method name
            backend: Optimization backend ('hyperopt', 'grid')

        Returns:
            Search space dictionary
        """
        if backend not in ["hyperopt", "grid"]:
            raise ValueError(f"Unsupported backend: {backend}")

        if method == "local_covariance":
            return SearchSpaceFactory._get_local_covariance_space(backend)
        elif method == "global_covariance":
            return SearchSpaceFactory._get_global_covariance_space(backend)
        elif method == "lasso":
            return SearchSpaceFactory._get_lasso_space(backend)
        elif method == "ridge":
            return SearchSpaceFactory._get_ridge_space(backend)
        elif method == "none":
            return {}
        else:
            raise ValueError(f"Unknown solver method: {method}")

    @staticmethod
    def get_lime_space(backend: str = "hyperopt") -> Dict[str, Any]:
        """Get search space for LIME parameters.

        Args:
            backend: Optimization backend ('hyperopt', 'grid')

        Returns:
            Search space dictionary
        """
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "num_samples": hp.choice("lime_num_samples", [100, 500, 1000, 2000]),
                "num_features": hp.choice("lime_num_features", [5, 10, 20, "auto"]),
                "feature_selection": hp.choice(
                    "lime_feature_selection",
                    ["forward_selection", "lasso_path", "none", "auto"],
                ),
            }
        elif backend == "grid":
            return {
                "num_samples": [100, 500, 1000, 2000],
                "num_features": [5, 10, 20, "auto"],
                "feature_selection": [
                    "forward_selection",
                    "lasso_path",
                    "none",
                    "auto",
                ],
            }

    @staticmethod
    def _get_lowrank_space(backend: str) -> Dict[str, Any]:
        """Get search space for lowrank simplification."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "n_components": hp.choice(
                    "lowrank_n_components", [0.8, 0.9, 0.95, 0.99]
                ),
                "whiten": hp.choice("lowrank_whiten", [True, False]),
                "svd_solver": hp.choice(
                    "lowrank_svd_solver", ["auto", "full", "arpack", "randomized"]
                ),
            }
        elif backend == "grid":
            return {
                "n_components": [0.8, 0.9, 0.95, 0.99],
                "whiten": [True, False],
                "svd_solver": ["auto", "full", "arpack", "randomized"],
            }

    @staticmethod
    def _get_superpixel_space(backend: str) -> Dict[str, Any]:
        """Get search space for superpixel simplification."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "n_segments": hp.choice("superpixel_n_segments", [50, 100, 200, 300]),
                "compactness": hp.uniform("superpixel_compactness", 0.1, 1.0),
                "sigma": hp.uniform("superpixel_sigma", 0.5, 2.0),
            }
        elif backend == "grid":
            return {
                "n_segments": [50, 100, 200, 300],
                "compactness": [0.1, 0.3, 0.5, 1.0],
                "sigma": [0.5, 1.0, 1.5, 2.0],
            }

    @staticmethod
    def _get_local_covariance_space(backend: str) -> Dict[str, Any]:
        """Get search space for local covariance solver."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "k_ratio": hp.uniform("local_cov_k_ratio", 0.05, 0.3),
                "bandwidth": hp.choice("local_cov_bandwidth", [None, "auto"]),
                "kernel": hp.choice(
                    "local_cov_kernel", ["gaussian", "exponential", "linear"]
                ),
                "shrinkage_intensity": hp.uniform("local_cov_shrinkage", 0.0, 0.5),
                "distance_metric": hp.choice(
                    "local_cov_distance", ["euclidean", "manhattan", "cosine"]
                ),
                "use_projection": hp.choice("local_cov_projection", [True, False]),
            }
        elif backend == "grid":
            return {
                "k_ratio": [0.05, 0.1, 0.15, 0.2, 0.3],
                "bandwidth": [None, "auto"],
                "kernel": ["gaussian", "exponential", "linear"],
                "shrinkage_intensity": [0.0, 0.1, 0.2, 0.5],
                "distance_metric": ["euclidean", "manhattan", "cosine"],
                "use_projection": [True, False],
            }

    @staticmethod
    def _get_global_covariance_space(backend: str) -> Dict[str, Any]:
        """Get search space for global covariance solver."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {"shrinkage_intensity": hp.uniform("global_cov_shrinkage", 0.0, 0.5)}
        elif backend == "grid":
            return {"shrinkage_intensity": [0.0, 0.1, 0.2, 0.3, 0.5]}

    @staticmethod
    def _get_lasso_space(backend: str) -> Dict[str, Any]:
        """Get search space for lasso solver."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "alpha": hp.loguniform("lasso_alpha", np.log(1e-4), np.log(1.0)),
                "max_iter": hp.choice("lasso_max_iter", [1000, 2000, 5000]),
                "selection": hp.choice("lasso_selection", ["cyclic", "random"]),
            }
        elif backend == "grid":
            return {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "max_iter": [1000, 2000, 5000],
                "selection": ["cyclic", "random"],
            }

    @staticmethod
    def _get_ridge_space(backend: str) -> Dict[str, Any]:
        """Get search space for ridge solver."""
        if backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt is required for hyperopt backend")
            return {
                "alpha": hp.loguniform("ridge_alpha", np.log(1e-4), np.log(10.0)),
                "solver": hp.choice(
                    "ridge_solver", ["auto", "svd", "cholesky", "lsqr"]
                ),
            }
        elif backend == "grid":
            return {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                "solver": ["auto", "svd", "cholesky", "lsqr"],
            }


def get_default_search_space(
    simplification: str = "lowrank",
    solver: str = "local_covariance",
    include_lime: bool = True,
    backend: str = "hyperopt",
) -> Dict[str, Any]:
    """Get default search space for PatternLocal optimization.

    Args:
        simplification: Simplification method name
        solver: Solver method name
        include_lime: Whether to include LIME parameters
        backend: Optimization backend

    Returns:
        Combined search space dictionary
    """
    search_space = {}

    # Add simplification parameters
    simp_space = SearchSpaceFactory.get_simplification_space(simplification, backend)
    for key, value in simp_space.items():
        search_space[f"simplification__{key}"] = value

    # Add solver parameters
    solver_space = SearchSpaceFactory.get_solver_space(solver, backend)
    for key, value in solver_space.items():
        search_space[f"solver__{key}"] = value

    # Add LIME parameters if requested
    if include_lime:
        lime_space = SearchSpaceFactory.get_lime_space(backend)
        for key, value in lime_space.items():
            search_space[f"lime__{key}"] = value

    return search_space


def create_custom_search_space(
    simplification_methods: List[str] = None,
    solver_methods: List[str] = None,
    include_lime: bool = True,
    backend: str = "hyperopt",
) -> Dict[str, Any]:
    """Create a custom search space that includes multiple methods.

    Args:
        simplification_methods: List of simplification methods to include
        solver_methods: List of solver methods to include
        include_lime: Whether to include LIME parameters
        backend: Optimization backend

    Returns:
        Search space with method selection
    """
    if backend == "hyperopt" and not HYPEROPT_AVAILABLE:
        raise ImportError("hyperopt is required for hyperopt backend")

    simplification_methods = simplification_methods or ["lowrank", "none"]
    solver_methods = solver_methods or ["local_covariance", "global_covariance"]

    search_space = {}

    # Add method selection
    if backend == "hyperopt":
        search_space["simplification_method"] = hp.choice(
            "simplification_method", simplification_methods
        )
        search_space["solver_method"] = hp.choice("solver_method", solver_methods)
    elif backend == "grid":
        search_space["simplification_method"] = simplification_methods
        search_space["solver_method"] = solver_methods

    # Add parameters for each method (conditional on selection for hyperopt)
    for simp_method in simplification_methods:
        simp_space = SearchSpaceFactory.get_simplification_space(simp_method, backend)
        for key, value in simp_space.items():
            param_name = f"simplification__{simp_method}__{key}"
            search_space[param_name] = value

    for solver_method in solver_methods:
        solver_space = SearchSpaceFactory.get_solver_space(solver_method, backend)
        for key, value in solver_space.items():
            param_name = f"solver__{solver_method}__{key}"
            search_space[param_name] = value

    # Add LIME parameters
    if include_lime:
        lime_space = SearchSpaceFactory.get_lime_space(backend)
        for key, value in lime_space.items():
            search_space[f"lime__{key}"] = value

    return search_space
