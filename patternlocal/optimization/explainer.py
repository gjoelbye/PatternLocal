"""
Optimized PatternLocal explainer with hyperparameter optimization capabilities.

This module provides the main interface for hyperparameter optimization
of PatternLocal explainers.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core.explainer import PatternLocalExplainer
from ..exceptions import OptimizationError
from .backends import create_backend
from .metrics import (
    create_fidelity_metric,
    evaluate_explanations_parallel,
)
from .search_spaces import get_default_search_space
from .utils import (
    OptimizationPaths,
    create_experiment_config,
    format_optimization_summary,
    load_optimization_results,
    parse_parameter_dict,
    save_experiment_info,
    save_optimization_results,
)

logger = logging.getLogger(__name__)

# Default batch sizes for evaluation
BATCH_SIZE = 10
SUBSET_SIZE = 50


class OptimizedPatternLocalExplainer(PatternLocalExplainer):
    """PatternLocal explainer with hyperparameter optimization capabilities.

    This class extends PatternLocalExplainer to provide automated hyperparameter
    optimization using various optimization backends and evaluation metrics.

    Examples:
        >>> # Basic optimization
        >>> explainer = OptimizedPatternLocalExplainer()
        >>> best_params = explainer.optimize_parameters(
        ...     X_val=X_val,
        ...     masks_val=masks_val,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     model=model
        ... )
        >>>
        >>> # Use optimized parameters
        >>> explainer.set_best_params(best_params)
        >>> explanation = explainer.explain_instance(instance, model.predict_proba, X_train)

        >>> # Custom optimization with specific metric
        >>> from patternlocal.optimization.metrics import create_fidelity_metric
        >>> metric_fn = create_fidelity_metric(n_samples=2000)
        >>> best_params = explainer.optimize_parameters(
        ...     X_val=X_val,
        ...     masks_val=masks_val,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     model=model,
        ...     metric_fn=metric_fn,
        ...     backend="hyperopt",
        ...     max_evals=200
        ... )
    """

    def __init__(
        self, cache_dir: str = "optimization_cache", n_jobs: int = -1, **kwargs
    ):
        """Initialize OptimizedPatternLocalExplainer.

        Args:
            cache_dir: Directory for caching optimization results
            n_jobs: Number of parallel jobs for optimization
            **kwargs: Arguments passed to PatternLocalExplainer
        """
        super().__init__(**kwargs)

        self.cache_dir = cache_dir
        self.n_jobs = n_jobs
        self.paths = OptimizationPaths(cache_dir)

        # Optimization state
        self.optimization_results = None
        self.best_params = None
        self.optimization_history = []

    def optimize_parameters(
        self,
        X_val: np.ndarray,
        masks_val: Optional[np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: Any,
        metric_fn: Optional[Callable] = None,
        search_space: Optional[Dict[str, Any]] = None,
        backend: str = "hyperopt",
        max_evals: int = 100,
        simplification: str = "lowrank",
        solver: str = "local_covariance",
        include_lime: bool = True,
        use_cache: bool = True,
        save_results: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize PatternLocal hyperparameters.

        Args:
            X_val: Validation instances
            masks_val: Ground truth importance masks (optional)
            X_train: Training data
            y_train: Training labels
            model: Trained model with predict_proba method
            metric_fn: Metric function to optimize (default: fidelity)
            search_space: Custom search space (optional)
            backend: Optimization backend ('hyperopt', 'grid', 'random')
            max_evals: Maximum number of evaluations
            simplification: Default simplification method
            solver: Default solver method
            include_lime: Whether to optimize LIME parameters
            use_cache: Whether to use cached results
            save_results: Whether to save results
            **kwargs: Additional arguments

        Returns:
            Dictionary with best parameters and optimization info
        """
        logger.info("Starting PatternLocal hyperparameter optimization")

        # Create experiment configuration for caching
        experiment_config = create_experiment_config(
            simplification=simplification,
            solver=solver,
            backend=backend,
            max_evals=max_evals,
            metric_name=metric_fn.__name__ if metric_fn else "fidelity",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            include_lime=include_lime,
            X_val_shape=X_val.shape,
            X_train_shape=X_train.shape,
            **kwargs,
        )

        # Check for cached results
        optimization_path = self.paths.get_optimization_path(experiment_config)
        if use_cache and os.path.exists(optimization_path):
            logger.info(f"Loading cached optimization results from {optimization_path}")
            try:
                cached_results = load_optimization_results(optimization_path)
                self.optimization_results = cached_results
                self.best_params = cached_results["best_params"]
                return cached_results["best_params"]
            except Exception as e:
                logger.warning(f"Failed to load cached results: {e}")

        # Create metric function if not provided
        if metric_fn is None:
            metric_fn = create_fidelity_metric(n_samples=1000)

        # Create search space if not provided
        if search_space is None:
            search_space = get_default_search_space(
                simplification=simplification,
                solver=solver,
                include_lime=include_lime,
                backend=backend,
            )

        # Create optimization backend
        optimization_backend = create_backend(backend, self.random_state)

        # Define objective function
        def objective(params):
            """Objective function for optimization."""
            return self._evaluate_parameters(
                params=params,
                X_val=X_val,
                masks_val=masks_val,
                X_train=X_train,
                y_train=y_train,
                model=model,
                metric_fn=metric_fn,
                simplification=simplification,
                solver=solver,
                **kwargs,
            )

        # Run optimization
        start_time = time.time()
        try:
            best_params, trials = optimization_backend.optimize(
                objective_fn=objective,
                search_space=search_space,
                max_evals=max_evals,
                **kwargs,
            )

            execution_time = time.time() - start_time

            # Format results
            self.optimization_history = optimization_backend.get_optimization_history()
            trials_info = {
                "n_evaluations": len(self.optimization_history),
                "n_successful": sum(
                    1 for h in self.optimization_history if h["status"] == "ok"
                ),
                "n_failed": sum(
                    1 for h in self.optimization_history if h["status"] != "ok"
                ),
                "convergence_history": [
                    h["loss"] for h in self.optimization_history if h["status"] == "ok"
                ],
            }

            optimization_summary = format_optimization_summary(
                best_params=best_params,
                best_loss=optimization_backend.best_loss,
                trials_info=trials_info,
                execution_time=execution_time,
            )

            # Store results
            self.optimization_results = {
                "best_params": best_params,
                "best_loss": optimization_backend.best_loss,
                "trials": trials,
                "optimization_history": self.optimization_history,
                "experiment_config": experiment_config,
                "summary": optimization_summary,
            }
            self.best_params = best_params

            # Save results if requested
            if save_results:
                try:
                    save_optimization_results(
                        self.optimization_results, optimization_path
                    )

                    # Save experiment info
                    info_path = self.paths.get_experiment_info_path(experiment_config)
                    save_experiment_info(
                        experiment_config, optimization_summary, info_path
                    )

                except Exception as e:
                    logger.warning(f"Failed to save optimization results: {e}")

            logger.info(
                f"Optimization completed in {execution_time:.2f}s. \
                    Best loss: {optimization_backend.best_loss:.6f}"
            )
            return best_params

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Hyperparameter optimization failed: {e}")

    def _evaluate_parameters(
        self,
        params: Dict[str, Any],
        X_val: np.ndarray,
        masks_val: Optional[np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: Any,
        metric_fn: Callable,
        simplification: str,
        solver: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a set of parameters.

        Args:
            params: Parameters to evaluate
            X_val: Validation data
            masks_val: Ground truth masks
            X_train: Training data
            y_train: Training labels
            model: Model instance
            metric_fn: Evaluation metric
            simplification: Base simplification method
            solver: Base solver method
            **kwargs: Additional arguments

        Returns:
            Evaluation result dictionary
        """
        try:
            # Parse parameters
            parsed_params = parse_parameter_dict(params)

            # Handle method selection if present
            current_simplification = simplification
            current_solver = solver
            if "method_selection" in parsed_params:
                current_simplification = parsed_params["method_selection"].get(
                    "simplification", simplification
                )
                current_solver = parsed_params["method_selection"].get("solver", solver)

            # Create explainer with specified parameters
            explainer = PatternLocalExplainer(
                simplification=current_simplification,
                solver=current_solver,
                lime_params=parsed_params.get("lime", {}),
                simplification_params=parsed_params.get("simplification", {}),
                solver_params=parsed_params.get("solver", {}),
                random_state=self.random_state,
            )

            # Fit explainer
            explainer.fit(X_train)

            # Evaluate on validation set
            errors, explanations, metadata = evaluate_explanations_parallel(
                explainer_instance=explainer,
                instances=X_val,
                masks=masks_val,
                X_train=X_train,
                y_train=y_train,
                predict_fn=model.predict_proba,
                metric_fn=metric_fn,
                n_jobs=self.n_jobs,
                batch_size=BATCH_SIZE,
                subset_size=SUBSET_SIZE,
                **kwargs,
            )

            # Calculate metrics
            valid_errors = [e for e in errors if e != float("inf")]
            if not valid_errors:
                return {
                    "loss": float("inf"),
                    "loss_std": float("inf"),
                    "status": "FAIL",
                }

            loss = np.mean(valid_errors)
            loss_std = np.std(valid_errors)

            return {
                "loss": float(loss),
                "loss_std": float(loss_std),
                "status": "OK",
                "metadata": metadata,
            }

        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return {
                "loss": float("inf"),
                "loss_std": float("inf"),
                "status": "FAIL",
                "error": str(e),
            }

    def set_best_params(
        self, best_params: Dict[str, Any]
    ) -> "OptimizedPatternLocalExplainer":
        """Set the explainer to use the best parameters found.

        Args:
            best_params: Best parameters from optimization

        Returns:
            Self for method chaining
        """
        parsed_params = parse_parameter_dict(best_params)

        # Update explainer components with best parameters
        if "simplification" in parsed_params and parsed_params["simplification"]:
            self.simplification_params.update(parsed_params["simplification"])

        if "solver" in parsed_params and parsed_params["solver"]:
            self.solver_params.update(parsed_params["solver"])

        if "lime" in parsed_params and parsed_params["lime"]:
            self.lime_params.update(parsed_params["lime"])

        # Recreate components with updated parameters
        self._initialize_components(
            self.simplification.__class__.__name__.replace(
                "Simplification", ""
            ).lower(),
            self.solver.__class__.__name__.replace("Solver", "").lower(),
        )

        # Mark as needing refit
        self.is_fitted = False
        self.best_params = best_params

        logger.info("Explainer updated with optimized parameters")
        return self

    def get_optimization_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of the last optimization run.

        Returns:
            Optimization summary or None if no optimization was run
        """
        if self.optimization_results is None:
            return None

        return self.optimization_results.get("summary")

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get detailed optimization history.

        Returns:
            List of optimization iterations with parameters and losses
        """
        return self.optimization_history.copy()

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history.

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt

            if not self.optimization_history:
                logger.warning("No optimization history to plot")
                return

            # Extract loss values
            losses = [
                h["loss"] for h in self.optimization_history if h["status"] == "ok"
            ]
            iterations = list(range(len(losses)))

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, losses, "b-", alpha=0.7, label="Loss")

            # Add best loss line
            if losses:
                best_loss = min(losses)
                plt.axhline(
                    y=best_loss,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Best Loss: {best_loss:.6f}",
                )

            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Hyperparameter Optimization History")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Optimization history plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")


def optimize_patternlocal(
    X_val: np.ndarray,
    masks_val: Optional[np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: Any,
    metric_fn: Optional[Callable] = None,
    backend: str = "hyperopt",
    max_evals: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """Convenient function for PatternLocal hyperparameter optimization.

    Args:
        X_val: Validation instances
        masks_val: Ground truth importance masks (optional)
        X_train: Training data
        y_train: Training labels
        model: Trained model
        metric_fn: Metric function (default: fidelity)
        backend: Optimization backend
        max_evals: Maximum evaluations
        **kwargs: Additional arguments

    Returns:
        Dictionary with optimization results
    """
    explainer = OptimizedPatternLocalExplainer(**kwargs)

    best_params = explainer.optimize_parameters(
        X_val=X_val,
        masks_val=masks_val,
        X_train=X_train,
        y_train=y_train,
        model=model,
        metric_fn=metric_fn,
        backend=backend,
        max_evals=max_evals,
        **kwargs,
    )

    return {
        "best_params": best_params,
        "explainer": explainer,
        "optimization_results": explainer.optimization_results,
    }
