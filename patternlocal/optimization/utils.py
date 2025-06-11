"""
Utilities for PatternLocal hyperparameter optimization.

This module provides utilities for saving, loading, and managing
optimization results and caching.
"""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OptimizationPaths:
    """Manages paths for optimization result caching."""

    def __init__(self, base_path: str = "optimization_cache"):
        """Initialize optimization paths.

        Args:
            base_path: Base directory for caching optimization results
        """
        self.base_path = base_path
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.base_path, exist_ok=True)

    def get_optimization_path(self, experiment_config: Dict[str, Any]) -> str:
        """Get path for optimization results based on configuration.

        Args:
            experiment_config: Configuration dictionary

        Returns:
            Path to optimization results file
        """
        # Create hash of configuration for unique filename
        config_str = json.dumps(experiment_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

        filename = f"optimization_{config_hash}.pkl"
        return os.path.join(self.base_path, filename)

    def get_experiment_info_path(self, experiment_config: Dict[str, Any]) -> str:
        """Get path for experiment information.

        Args:
            experiment_config: Configuration dictionary

        Returns:
            Path to experiment info file
        """
        config_str = json.dumps(experiment_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

        filename = f"experiment_info_{config_hash}.json"
        return os.path.join(self.base_path, filename)


def save_optimization_results(
    results: Dict[str, Any], file_path: str, include_metadata: bool = True
) -> None:
    """Save optimization results to file.

    Args:
        results: Optimization results dictionary
        file_path: Path to save results
        include_metadata: Whether to include metadata
    """
    try:
        # Add metadata if requested
        if include_metadata:
            results = results.copy()
            results["metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0",
            }

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save results
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Optimization results saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")
        raise


def load_optimization_results(file_path: str) -> Dict[str, Any]:
    """Load optimization results from file.

    Args:
        file_path: Path to results file

    Returns:
        Loaded optimization results
    """
    try:
        with open(file_path, "rb") as f:
            results = pickle.load(f)

        logger.info(f"Optimization results loaded from {file_path}")
        return results

    except Exception as e:
        logger.error(f"Failed to load optimization results: {e}")
        raise


def save_experiment_info(
    config: Dict[str, Any], results_summary: Dict[str, Any], file_path: str
) -> None:
    """Save experiment information in human-readable format.

    Args:
        config: Experiment configuration
        results_summary: Summary of results
        file_path: Path to save info
    """
    try:
        experiment_info = {
            "config": config,
            "results_summary": results_summary,
            "created_at": datetime.now().isoformat(),
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save as JSON
        with open(file_path, "w") as f:
            json.dump(experiment_info, f, indent=2, default=str)

        logger.info(f"Experiment info saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save experiment info: {e}")
        raise


def create_experiment_config(
    simplification: str = "lowrank",
    solver: str = "local_covariance",
    backend: str = "hyperopt",
    max_evals: int = 100,
    metric_name: str = "fidelity",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create standardized experiment configuration.

    Args:
        simplification: Simplification method
        solver: Solver method
        backend: Optimization backend
        max_evals: Maximum evaluations
        metric_name: Metric to optimize
        n_jobs: Number of parallel jobs
        random_state: Random seed
        **kwargs: Additional configuration

    Returns:
        Experiment configuration dictionary
    """
    config = {
        "simplification": simplification,
        "solver": solver,
        "backend": backend,
        "max_evals": max_evals,
        "metric_name": metric_name,
        "n_jobs": n_jobs,
        "random_state": random_state,
        **kwargs,
    }

    return config


def parse_parameter_dict(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse flattened parameter dictionary into component dictionaries.

    Args:
        params: Flattened parameter dictionary with keys like 'component__param'

    Returns:
        Dictionary with component names as keys
    """
    parsed = {"simplification": {}, "solver": {}, "lime": {}}

    for key, value in params.items():
        if "__" in key:
            component, param_name = key.split("__", 1)
            if component in parsed:
                parsed[component][param_name] = value
            else:
                # Handle nested parameters like 'solver__local_covariance__k_ratio'
                parts = key.split("__")
                if len(parts) >= 2:
                    component = parts[0]
                    if component in parsed:
                        nested_key = "__".join(parts[1:])
                        parsed[component][nested_key] = value
        else:
            # Parameters without component prefix (e.g., method selection)
            if "simplification_method" in key:
                parsed["method_selection"] = parsed.get("method_selection", {})
                parsed["method_selection"]["simplification"] = value
            elif "solver_method" in key:
                parsed["method_selection"] = parsed.get("method_selection", {})
                parsed["method_selection"]["solver"] = value

    return parsed


def format_optimization_summary(
    best_params: Dict[str, Any],
    best_loss: float,
    trials_info: Dict[str, Any],
    execution_time: float,
) -> Dict[str, Any]:
    """Format optimization results summary.

    Args:
        best_params: Best parameters found
        best_loss: Best loss value
        trials_info: Information about trials/evaluations
        execution_time: Total execution time in seconds

    Returns:
        Formatted summary dictionary
    """
    summary = {
        "best_loss": float(best_loss),
        "best_params": best_params,
        "optimization_stats": {
            "total_evaluations": trials_info.get("n_evaluations", 0),
            "successful_evaluations": trials_info.get("n_successful", 0),
            "failed_evaluations": trials_info.get("n_failed", 0),
            "execution_time_seconds": float(execution_time),
            "average_time_per_evaluation": float(
                execution_time / max(1, trials_info.get("n_evaluations", 1))
            ),
        },
    }

    # Add convergence information if available
    if "convergence_history" in trials_info:
        history = trials_info["convergence_history"]
        summary["convergence"] = {
            "initial_loss": history[0] if history else None,
            "final_loss": history[-1] if history else None,
            "improvement": (history[0] - history[-1]) if len(history) > 1 else 0.0,
            "convergence_iteration": _find_convergence_point(history),
        }

    return summary


def _find_convergence_point(
    loss_history: list, patience: int = 10, min_improvement: float = 1e-6
) -> Optional[int]:
    """Find point where optimization converged.

    Args:
        loss_history: List of loss values over iterations
        patience: Number of iterations without improvement to consider converged
        min_improvement: Minimum improvement to consider significant

    Returns:
        Iteration where convergence occurred, or None
    """
    if len(loss_history) < patience + 1:
        return None

    best_loss = float("inf")
    iterations_without_improvement = 0

    for i, loss in enumerate(loss_history):
        if loss < best_loss - min_improvement:
            best_loss = loss
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

            if iterations_without_improvement >= patience:
                return i - patience

    return None


def cleanup_old_cache_files(
    cache_dir: str, max_age_days: int = 30, dry_run: bool = False
) -> list:
    """Clean up old cache files.

    Args:
        cache_dir: Cache directory path
        max_age_days: Maximum age in days before deletion
        dry_run: If True, only return files that would be deleted

    Returns:
        List of files that were (or would be) deleted
    """
    import time

    if not os.path.exists(cache_dir):
        return []

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    deleted_files = []

    try:
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)

            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > max_age_seconds:
                    if not dry_run:
                        os.remove(file_path)
                        logger.info(f"Deleted old cache file: {filename}")
                    deleted_files.append(filename)

        if dry_run and deleted_files:
            logger.info(f"Would delete {len(deleted_files)} old cache files")
        elif deleted_files:
            logger.info(f"Deleted {len(deleted_files)} old cache files")

    except Exception as e:
        logger.error(f"Error cleaning up cache files: {e}")

    return deleted_files


def get_cache_info(cache_dir: str) -> Dict[str, Any]:
    """Get information about cached optimization results.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dictionary with cache information
    """
    if not os.path.exists(cache_dir):
        return {"exists": False}

    try:
        files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)

        return {
            "exists": True,
            "num_cached_results": len(files),
            "total_size_bytes": total_size,
            "cache_dir": cache_dir,
            "files": files,
        }

    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return {"exists": True, "error": str(e)}
