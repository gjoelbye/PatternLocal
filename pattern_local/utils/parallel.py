"""
Parallel processing utilities for batch operations.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Utilities for parallel processing of batch operations."""

    def __init__(
        self,
        n_jobs: int = 1,
        backend: str = "threading",
        max_workers: Optional[int] = None,
    ):
        """Initialize parallel processor.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            backend: 'threading' or 'multiprocessing'
            max_workers: Maximum number of workers (overrides n_jobs if provided)
        """
        self.backend = backend

        if max_workers is not None:
            self.n_jobs = max_workers
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = max(1, n_jobs)

        if self.n_jobs == 1:
            self.backend = "sequential"  # Force sequential for single job

    def map(self, func: Callable, iterable: List[Any], *args, **kwargs) -> List[Any]:
        """Apply function to each item in iterable in parallel.

        Args:
            func: Function to apply
            iterable: Items to process
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func

        Returns:
            List of results
        """
        if self.backend == "sequential" or len(iterable) == 1:
            return [func(item, *args, **kwargs) for item in iterable]

        # Prepare function with additional arguments
        if args or kwargs:
            func_with_args = partial(func, *args, **kwargs)
        else:
            func_with_args = func

        if self.backend == "threading":
            return self._map_threading(func_with_args, iterable)
        elif self.backend == "multiprocessing":
            return self._map_multiprocessing(func_with_args, iterable)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _map_threading(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Map using thread pool executor."""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            results = []

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    raise

            # Maintain original order
            future_to_index = {
                executor.submit(func, item): i for i, item in enumerate(iterable)
            }
            ordered_results = [None] * len(iterable)

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(func, item) for item in iterable]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        index = future_to_index[future]
                        ordered_results[index] = result
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
                        raise

            return ordered_results

    def _map_multiprocessing(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Map using process pool executor."""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            results = []

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    raise

            # Maintain original order
            future_to_index = {
                executor.submit(func, item): i for i, item in enumerate(iterable)
            }
            ordered_results = [None] * len(iterable)

            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(func, item) for item in iterable]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        index = future_to_index[future]
                        ordered_results[index] = result
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
                        raise

            return ordered_results

    def batch_process(
        self, func: Callable, data: np.ndarray, batch_size: int = 32, *args, **kwargs
    ) -> List[Any]:
        """Process data in batches with parallel execution.

        Args:
            func: Function to apply to each batch
            data: Data to process (first dimension is batch dimension)
            batch_size: Size of each batch
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func

        Returns:
            List of results from each batch
        """
        n_samples = data.shape[0]
        batches = []

        # Create batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]
            batches.append(batch)

        # Process batches in parallel
        return self.map(func, batches, *args, **kwargs)


def parallel_explain_instances(
    explainer,
    instances: np.ndarray,
    predict_fn: Callable,
    X_train: np.ndarray,
    n_jobs: int = 1,
    backend: str = "threading",
    **kwargs,
) -> List[dict]:
    """Explain multiple instances in parallel.

    Args:
        explainer: Fitted PatternLocalExplainer instance
        instances: Instances to explain, shape (n_instances, n_features)
        predict_fn: Prediction function
        X_train: Training data
        n_jobs: Number of parallel jobs
        backend: 'threading' or 'multiprocessing'
        **kwargs: Additional arguments for explain_instance

    Returns:
        List of explanation dictionaries
    """
    if not explainer.is_fitted:
        raise ValueError("Explainer must be fitted before explaining instances")

    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)

    def explain_single_instance(instance):
        return explainer.explain_instance(instance, predict_fn, X_train, **kwargs)

    # Convert to list of instances
    instance_list = [instances[i] for i in range(instances.shape[0])]

    return processor.map(explain_single_instance, instance_list)


def parallel_fit_simplifications(
    simplification_configs: List[dict],
    X_train: np.ndarray,
    n_jobs: int = 1,
    backend: str = "threading",
    **kwargs,
) -> List[Any]:
    """Fit multiple simplification methods in parallel.

    Args:
        simplification_configs: List of simplification configurations
        X_train: Training data
        n_jobs: Number of parallel jobs
        backend: 'threading' or 'multiprocessing'
        **kwargs: Additional fitting arguments

    Returns:
        List of fitted simplification objects
    """
    from ..simplification.registry import SimplificationRegistry

    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)

    def fit_simplification(config):
        method = config["method"]
        params = config.get("params", {})

        simplification = SimplificationRegistry.create(method, params)
        return simplification.fit(X_train, **kwargs)

    return processor.map(fit_simplification, simplification_configs)


def parallel_solver_comparison(
    solver_configs: List[dict],
    lime_weights: np.ndarray,
    lime_intercept: float,
    instance: np.ndarray,
    X_train: np.ndarray,
    n_jobs: int = 1,
    backend: str = "threading",
) -> List[np.ndarray]:
    """Compare multiple solvers in parallel.

    Args:
        solver_configs: List of solver configurations
        lime_weights: LIME weights
        lime_intercept: LIME intercept
        instance: Instance being explained
        X_train: Training data
        n_jobs: Number of parallel jobs
        backend: 'threading' or 'multiprocessing'

    Returns:
        List of pattern weights from each solver
    """
    from ..solvers.registry import SolverRegistry

    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)

    def solve_with_config(config):
        method = config["method"]
        params = config.get("params", {})

        solver = SolverRegistry.create(method, params)
        return solver.solve(lime_weights, lime_intercept, instance, X_train)

    return processor.map(solve_with_config, solver_configs)


def parallel_cross_validation(
    explainer_configs: List[dict],
    X_train: np.ndarray,
    X_test: np.ndarray,
    predict_fn: Callable,
    cv_folds: int = 5,
    n_jobs: int = 1,
    backend: str = "threading",
    **kwargs,
) -> List[dict]:
    """Perform cross-validation of explainer configurations in parallel.

    Args:
        explainer_configs: List of explainer configurations
        X_train: Training data
        X_test: Test data for explanation
        predict_fn: Prediction function
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        backend: 'threading' or 'multiprocessing'
        **kwargs: Additional arguments for explain_instance

    Returns:
        List of cross-validation results
    """
    from sklearn.model_selection import KFold

    from ..core.explainer import PatternLocalExplainer

    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def evaluate_config(config):
        fold_results = []

        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]

            # Create and fit explainer
            explainer = PatternLocalExplainer(**config)
            explainer.fit(X_fold_train)

            # Evaluate on validation set
            fold_explanations = []
            for instance in X_fold_val[
                : min(10, len(X_fold_val))
            ]:  # Limit for efficiency
                explanation = explainer.explain_instance(
                    instance, predict_fn, X_fold_train, **kwargs
                )
                fold_explanations.append(explanation)

            fold_results.append(
                {"fold_explanations": fold_explanations, "fold_size": len(X_fold_val)}
            )

        return {"config": config, "fold_results": fold_results}

    return processor.map(evaluate_config, explainer_configs)


def auto_select_backend(n_jobs: int, task_type: str = "computation") -> str:
    """Automatically select the best backend for parallel processing.

    Args:
        n_jobs: Number of parallel jobs
        task_type: Type of task ('computation', 'io', 'mixed')

    Returns:
        Recommended backend ('threading', 'multiprocessing', or 'sequential')
    """
    if n_jobs == 1:
        return "sequential"

    # For CPU-intensive tasks, use multiprocessing
    if task_type == "computation":
        return "multiprocessing"

    # For I/O-intensive tasks, use threading
    elif task_type == "io":
        return "threading"

    # For mixed tasks, use threading for simplicity
    else:
        return "threading"


def estimate_optimal_batch_size(
    data_size: int, memory_limit_mb: int = 1024, item_size_bytes: int = None
) -> int:
    """Estimate optimal batch size for parallel processing.

    Args:
        data_size: Number of items to process
        memory_limit_mb: Memory limit in MB
        item_size_bytes: Estimated size of each item in bytes

    Returns:
        Recommended batch size
    """
    if item_size_bytes is None:
        # Default estimate for typical ML data
        item_size_bytes = 1024  # 1KB per item

    memory_limit_bytes = memory_limit_mb * 1024 * 1024
    max_batch_size = memory_limit_bytes // item_size_bytes

    # Ensure reasonable bounds
    min_batch_size = 1
    max_reasonable_batch_size = min(data_size, 1000)

    batch_size = min(max_batch_size, max_reasonable_batch_size)
    batch_size = max(batch_size, min_batch_size)

    return batch_size
