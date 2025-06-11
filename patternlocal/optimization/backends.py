"""
Optimization backends for PatternLocal hyperparameter optimization.

This module provides different optimization algorithms that can be used
to optimize PatternLocal hyperparameters.
"""

import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from hyperopt import Trials, fmin, tpe

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationBackend(ABC):
    """Abstract base class for optimization backends."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize optimization backend.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.trials = None
        self.best_params = None
        self.best_loss = None

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        max_evals: int = 100,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run optimization.

        Args:
            objective_fn: Objective function to minimize
            search_space: Parameter search space
            max_evals: Maximum number of evaluations
            **kwargs: Additional backend-specific arguments

        Returns:
            Tuple of (best_params, trials/results)
        """
        pass

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history.

        Returns:
            List of evaluation results
        """
        if self.trials is None:
            return []
        return self._extract_history()

    @abstractmethod
    def _extract_history(self) -> List[Dict[str, Any]]:
        """Extract optimization history from trials."""
        pass


class HyperoptBackend(OptimizationBackend):
    """Hyperopt-based optimization backend using TPE algorithm."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize HyperoptBackend.

        Args:
            random_state: Random seed for reproducibility
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for HyperoptBackend")

        super().__init__(random_state)
        self.algorithm = tpe.suggest

    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        max_evals: int = 100,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run hyperopt optimization.

        Args:
            objective_fn: Objective function to minimize
            search_space: Hyperopt search space
            max_evals: Maximum number of evaluations
            **kwargs: Additional hyperopt arguments

        Returns:
            Tuple of (best_params, trials)
        """
        logger.info(f"Starting hyperopt optimization with {max_evals} evaluations")

        # Create trials object
        self.trials = Trials()

        # Set random state if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Run optimization
        try:
            best_params = fmin(
                fn=objective_fn,
                space=search_space,
                algo=self.algorithm,
                max_evals=max_evals,
                trials=self.trials,
                rstate=(
                    np.random.RandomState(self.random_state)
                    if self.random_state
                    else None
                ),
                **kwargs,
            )

            # Get best loss
            self.best_loss = min(
                [trial["result"]["loss"] for trial in self.trials.trials]
            )
            self.best_params = best_params

            logger.info(f"Optimization completed. Best loss: {self.best_loss}")
            return best_params, self.trials

        except Exception as e:
            logger.error(f"Hyperopt optimization failed: {e}")
            raise

    def _extract_history(self) -> List[Dict[str, Any]]:
        """Extract optimization history from hyperopt trials."""
        history = []
        for i, trial in enumerate(self.trials.trials):
            result = trial["result"]
            params = trial["misc"]["vals"]

            # Flatten parameter values
            flat_params = {}
            for key, value_list in params.items():
                if value_list:  # Check if list is not empty
                    flat_params[key] = value_list[0]

            history.append(
                {
                    "iteration": i,
                    "params": flat_params,
                    "loss": result.get("loss", float("inf")),
                    "loss_std": result.get("loss_std", 0.0),
                    "status": result.get("status", "unknown"),
                }
            )

        return history


class GridSearchBackend(OptimizationBackend):
    """Grid search optimization backend."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize GridSearchBackend.

        Args:
            random_state: Random seed for reproducibility (affects grid ordering)
        """
        super().__init__(random_state)
        self.results = []

    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        max_evals: int = 100,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run grid search optimization.

        Args:
            objective_fn: Objective function to minimize
            search_space: Grid search space (dict with lists of values)
            max_evals: Maximum number of evaluations (limits grid size)
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (best_params, results)
        """
        logger.info("Starting grid search optimization")

        # Generate parameter combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]

        # Create grid
        param_combinations = list(product(*param_values))

        # Limit combinations if needed
        if len(param_combinations) > max_evals:
            logger.warning(
                f"Grid has {len(param_combinations)} combinations, "
                f"limiting to {max_evals} evaluations"
            )
            if self.random_state is not None:
                np.random.seed(self.random_state)
                indices = np.random.choice(
                    len(param_combinations), max_evals, replace=False
                )
                param_combinations = [param_combinations[i] for i in indices]
            else:
                param_combinations = param_combinations[:max_evals]

        # Evaluate each combination
        self.results = []
        best_loss = float("inf")
        best_params = None

        for i, param_values in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_values))

            try:
                # Evaluate objective function
                result = objective_fn(params)
                loss = result["loss"] if isinstance(result, dict) else result
                loss_std = (
                    result.get("loss_std", 0.0) if isinstance(result, dict) else 0.0
                )

                # Store result
                self.results.append(
                    {
                        "iteration": i,
                        "params": params.copy(),
                        "loss": loss,
                        "loss_std": loss_std,
                        "status": "ok",
                    }
                )

                # Update best
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                logger.debug(f"Iteration {i}: loss={loss:.6f}, params={params}")

            except Exception as e:
                logger.warning(f"Evaluation {i} failed: {e}")
                self.results.append(
                    {
                        "iteration": i,
                        "params": params.copy(),
                        "loss": float("inf"),
                        "loss_std": float("inf"),
                        "status": "failed",
                    }
                )

        self.best_loss = best_loss
        self.best_params = best_params
        self.trials = self.results  # For compatibility

        logger.info(f"Grid search completed. Best loss: {best_loss}")
        return best_params, self.results

    def _extract_history(self) -> List[Dict[str, Any]]:
        """Extract optimization history from grid search results."""
        return self.results.copy()


class RandomSearchBackend(OptimizationBackend):
    """Random search optimization backend."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize RandomSearchBackend.

        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.results = []

    def optimize(
        self,
        objective_fn: Callable,
        search_space: Dict[str, Any],
        max_evals: int = 100,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run random search optimization.

        Args:
            objective_fn: Objective function to minimize
            search_space: Search space (dict with lists of values)
            max_evals: Number of random evaluations
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (best_params, results)
        """
        logger.info(f"Starting random search optimization with {max_evals} evaluations")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.results = []
        best_loss = float("inf")
        best_params = None

        for i in range(max_evals):
            # Sample random parameters
            params = {}
            for name, values in search_space.items():
                if isinstance(values, list):
                    params[name] = np.random.choice(values)
                else:
                    # Assume it's a continuous range [min, max]
                    params[name] = np.random.uniform(values[0], values[1])

            try:
                # Evaluate objective function
                result = objective_fn(params)
                loss = result["loss"] if isinstance(result, dict) else result
                loss_std = (
                    result.get("loss_std", 0.0) if isinstance(result, dict) else 0.0
                )

                # Store result
                self.results.append(
                    {
                        "iteration": i,
                        "params": params.copy(),
                        "loss": loss,
                        "loss_std": loss_std,
                        "status": "ok",
                    }
                )

                # Update best
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                logger.debug(f"Iteration {i}: loss={loss:.6f}, params={params}")

            except Exception as e:
                logger.warning(f"Evaluation {i} failed: {e}")
                self.results.append(
                    {
                        "iteration": i,
                        "params": params.copy(),
                        "loss": float("inf"),
                        "loss_std": float("inf"),
                        "status": "failed",
                    }
                )

        self.best_loss = best_loss
        self.best_params = best_params
        self.trials = self.results  # For compatibility

        logger.info(f"Random search completed. Best loss: {best_loss}")
        return best_params, self.results

    def _extract_history(self) -> List[Dict[str, Any]]:
        """Extract optimization history from random search results."""
        return self.results.copy()


def create_backend(
    backend_name: str, random_state: Optional[int] = None
) -> OptimizationBackend:
    """Create an optimization backend.

    Args:
        backend_name: Name of the backend ('hyperopt', 'grid', 'random')
        random_state: Random seed for reproducibility

    Returns:
        Optimization backend instance
    """
    if backend_name == "hyperopt":
        return HyperoptBackend(random_state)
    elif backend_name == "grid":
        return GridSearchBackend(random_state)
    elif backend_name == "random":
        return RandomSearchBackend(random_state)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
