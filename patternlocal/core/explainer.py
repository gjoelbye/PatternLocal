"""
Main PatternLocal explainer class with enhanced architecture.
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from ..config.config import ExplainerConfig
from ..config.validation import (
    ParameterValidator,
    validate_array_input,
    validate_fitted,
)
from ..core.strategies import StrategyFactory
from ..exceptions import (
    ConfigurationError,
    ExplanationError,
    FittingError,
)
from ..simplification.base import BaseSimplification
from ..simplification.registry import SimplificationRegistry
from ..solvers.base import BaseSolver
from ..solvers.registry import SolverRegistry

logger = logging.getLogger(__name__)


class PatternLocalExplainer:
    """PatternLocal explainer.

    This class provides a unified interface for generating local explanations
    using patternlocal methods with various feature simplification and solver options.
    Supports both tabular and image data through a unified API with enhanced
    features like configuration management and fluent interface.

    The explainer works in three stages:
    1. Simplification: Transform data to a different representation (optional)
    2. LIME: Generate local linear explanation (tabular or image mode)
    3. PatternLocal: Compute patternlocal weights from LIME weights using various solvers

    Examples:
        >>> # Basic usage (backward compatible)
        >>> explainer = PatternLocalExplainer(
        ...     simplification='lowrank',
        ...     solver='local_covariance'
        ... )
        >>> explainer.fit(X_train)
        >>> explanation = explainer.explain_instance(instance, predict_fn, X_train)

        >>> # Fluent interface
        >>> explainer = (PatternLocalExplainer()
        ...              .with_simplification('lowrank', n_components=10)
        ...              .with_solver('local_covariance', k_ratio=0.1)
        ...              .fit(X_train))

        >>> # Configuration from dict
        >>> config = {'simplification': 'lowrank', 'solver': 'local_covariance'}
        >>> explainer = PatternLocalExplainer.from_config(config)
    """

    def __init__(
        self,
        simplification: Union[str, BaseSimplification] = "none",
        solver: Union[str, BaseSolver] = "local_covariance",
        lime_params: Optional[Dict[str, Any]] = None,
        simplification_params: Optional[Dict[str, Any]] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize PatternLocal explainer.

        Args:
            simplification: Simplification method or instance
            solver: PatternLocal solver or instance
            lime_params: Parameters for LIME explainer
            simplification_params: Parameters for simplification method
            solver_params: Parameters for patternlocal solver
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        self.random_state = random_state

        # Store parameters for backward compatibility
        self.lime_params = lime_params or {}
        self.simplification_params = simplification_params or {}
        self.solver_params = solver_params or {}

        # Initialize components
        self._initialize_components(simplification, solver)

        # Internal state
        self._lime_explainer = None
        self._X_train_simplified = None
        self.is_fitted = False

    def _initialize_components(
        self,
        simplification: Union[str, BaseSimplification],
        solver: Union[str, BaseSolver],
    ) -> None:
        """Initialize simplification and solver components."""
        self.logger.info("Initializing PatternLocalExplainer components")

        # Create simplification method
        if isinstance(simplification, BaseSimplification):
            self.simplification = simplification
        elif isinstance(simplification, str):
            try:
                self.simplification = SimplificationRegistry.create(
                    simplification, self.simplification_params
                )
            except Exception:
                # Fallback to original method for backward compatibility
                self.simplification = self._create_simplification_legacy(
                    simplification, self.simplification_params
                )
        else:
            raise ConfigurationError(f"Invalid simplification: {simplification}")

        # Create solver
        if isinstance(solver, BaseSolver):
            self.solver = solver
        elif isinstance(solver, str):
            try:
                self.solver = SolverRegistry.create(solver, self.solver_params)
            except Exception:
                # Fallback to original method for backward compatibility
                self.solver = self._create_solver_legacy(solver, self.solver_params)
        else:
            raise ConfigurationError(f"Invalid solver: {solver}")

        # Auto-detect and create strategy
        self._strategy = StrategyFactory.auto_detect_strategy(
            self.simplification, self.lime_params
        )

        self.logger.info(
            f"Initialized with simplification: {
                type(
                    self.simplification).__name__}, "
            f"solver: {
                type(
                    self.solver).__name__}, "
            f"mode: {
                        self.mode}"
        )

    def _create_simplification_legacy(
        self, simplification: str, params: Optional[Dict[str, Any]]
    ) -> BaseSimplification:
        """Legacy simplification creation for backward compatibility."""
        from ..simplification import (
            LowRankSimplification,
            NoSimplification,
            SuperpixelSimplification,
        )

        params = params or {}

        if simplification == "none":
            return NoSimplification(params)
        elif simplification == "lowrank":
            return LowRankSimplification(params)
        elif simplification == "superpixel":
            return SuperpixelSimplification(params)
        else:
            raise ConfigurationError(f"Unknown simplification method: {simplification}")

    def _create_solver_legacy(
        self, solver: str, params: Optional[Dict[str, Any]]
    ) -> BaseSolver:
        """Legacy solver creation for backward compatibility."""
        from ..solvers import (
            GlobalCovarianceSolver,
            LassoSolver,
            LocalCovarianceSolver,
            NoSolver,
            RidgeSolver,
        )

        params = params or {}

        if solver == "none":
            return NoSolver(params)
        elif solver == "global_covariance":
            return GlobalCovarianceSolver(params)
        elif solver == "local_covariance":
            return LocalCovarianceSolver(params)
        elif solver == "lasso":
            return LassoSolver(params)
        elif solver == "ridge":
            return RidgeSolver(params)
        else:
            raise ConfigurationError(f"Unknown solver: {solver}")

    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], ExplainerConfig]
    ) -> "PatternLocalExplainer":
        """Create explainer from configuration.

        Args:
            config: Configuration dictionary or ExplainerConfig object

        Returns:
            Configured explainer instance
        """
        if isinstance(config, dict):
            return cls(**config)
        elif isinstance(config, ExplainerConfig):
            return cls(**config.to_dict())
        else:
            raise ConfigurationError(f"Invalid config type: {type(config)}")

    def with_simplification(
        self, method: str, **params: Any
    ) -> "PatternLocalExplainer":
        """Set simplification method using fluent interface.

        Args:
            method: Simplification method name
            **params: Method parameters

        Returns:
            Self for method chaining
        """
        try:
            self.simplification = SimplificationRegistry.create(method, params)
        except Exception:
            self.simplification = self._create_simplification_legacy(method, params)

        # Update strategy
        self._strategy = StrategyFactory.auto_detect_strategy(
            self.simplification, self.lime_params
        )

        return self

    def with_solver(self, method: str, **params: Any) -> "PatternLocalExplainer":
        """Set solver using fluent interface.

        Args:
            method: Solver method name
            **params: Solver parameters

        Returns:
            Self for method chaining
        """
        try:
            self.solver = SolverRegistry.create(method, params)
        except Exception:
            self.solver = self._create_solver_legacy(method, params)

        return self

    def with_lime_params(self, **params: Any) -> "PatternLocalExplainer":
        """Set LIME parameters using fluent interface.

        Args:
            **params: LIME parameters

        Returns:
            Self for method chaining
        """
        self.lime_params.update(params)

        # Update strategy
        self._strategy = StrategyFactory.auto_detect_strategy(
            self.simplification, self.lime_params
        )

        return self

    @validate_array_input
    def fit(self, X_train: np.ndarray, **kwargs: Any) -> "PatternLocalExplainer":
        """Fit the explainer to training data.

        Args:
            X_train: Training data, shape (n_samples, n_features)
            **kwargs: Additional arguments for simplification fitting
                - image_shape: Required for image mode (height, width)

        Returns:
            Self for method chaining

        Raises:
            FittingError: If fitting fails
            ValidationError: If inputs are invalid
        """
        try:
            self.logger.info(
                f"Fitting explainer on {
                    X_train.shape[0]} samples"
            )

            # Validate training data
            ParameterValidator.validate_training_data(X_train)

            # Fit simplification method
            self.simplification.fit(X_train, **kwargs)

            # Transform training data
            self._X_train_simplified = self.simplification.transform_training_data(
                X_train
            )

            # Create LIME explainer
            self._lime_explainer = self._strategy.create_explainer(
                self._X_train_simplified, self.lime_params, self.random_state
            )

            self.is_fitted = True
            self.logger.info("Fitting completed successfully")
            return self

        except Exception as e:
            raise FittingError(f"Failed to fit explainer: {e}")

    @validate_fitted
    @validate_array_input
    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        X_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Explain a single instance.

        Args:
            instance: Instance to explain, shape (n_features,)
            predict_fn: Model prediction function
            X_train: Training data (optional, uses fitted data if not provided)
            **kwargs: Additional arguments
                - num_samples: Number of samples for LIME
                - num_features: Number of features to show in explanation
                - labels: Labels for image explanation (required for image mode)

        Returns:
            Dictionary containing:
                - 'pattern_weights': Pattern explanation weights
                - 'lime_weights': Original LIME weights
                - 'lime_intercept': LIME intercept
                - 'local_exp': LIME explanation object (if available)
                - 'metadata': Additional metadata

        Raises:
            ExplanationError: If explanation generation fails
            ValidationError: If inputs are invalid
        """
        try:
            # Use provided X_train or fitted data
            if X_train is None:
                X_train_simplified = self._X_train_simplified
            else:
                ParameterValidator.validate_array(X_train, "X_train", ndim=2)
                X_train_simplified = self.simplification.transform_training_data(
                    X_train
                )

            # Transform instance
            instance_simplified = self.simplification.transform_instance(instance)

            # Create prediction function for simplified space
            predict_fn_simplified = self.simplification.create_predict_function(
                predict_fn
            )

            # Generate LIME explanation using strategy
            explanation = self._strategy.generate_explanation(
                self._lime_explainer,
                instance,
                instance_simplified,
                predict_fn_simplified,
                self.simplification,
                **kwargs,
            )

            # Extract LIME weights and intercept
            lime_weights, lime_intercept = self._strategy.extract_explanation(
                explanation, instance_simplified.shape[0]
            )

            # Compute pattern weights using solver
            pattern_weights_simplified = self.solver.solve(
                lime_weights=lime_weights,
                lime_intercept=lime_intercept,
                instance=instance_simplified,
                X_train=X_train_simplified,
            )

            # Transform weights back to original space
            pattern_weights = self.simplification.inverse_transform_weights(
                pattern_weights_simplified
            )
            lime_weights_original = self.simplification.inverse_transform_weights(
                lime_weights
            )

            return {
                "pattern_weights": pattern_weights,
                "lime_weights": lime_weights_original,
                "lime_intercept": lime_intercept,
                "local_exp": explanation,
                "metadata": {
                    "simplification_method": type(self.simplification).__name__,
                    "solver_method": type(self.solver).__name__,
                    "lime_mode": self.mode,
                    "instance_shape": instance.shape,
                    "simplified_shape": instance_simplified.shape,
                },
            }

        except Exception as e:
            raise ExplanationError(f"Failed to explain instance: {e}")

    @property
    def mode(self) -> str:
        """Current LIME mode (tabular or image)."""
        return self._strategy.detect_mode(self.simplification, self.lime_params)

    @property
    def simplification_method(self) -> str:
        """Name of the current simplification method."""
        return type(self.simplification).__name__

    @property
    def solver_method(self) -> str:
        """Name of the current solver method."""
        return type(self.solver).__name__

    def get_explainer_info(self) -> Dict[str, Any]:
        """Get information about the explainer.

        Returns:
            Dictionary with explainer information
        """
        return {
            "simplification": {
                "type": type(self.simplification).__name__,
                "params": getattr(self.simplification, "params", {}),
                "is_fitted": getattr(self.simplification, "is_fitted", False),
            },
            "solver": {
                "type": type(self.solver).__name__,
                "params": getattr(self.solver, "params", {}),
            },
            "lime_mode": self.mode,
            "is_fitted": self.is_fitted,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"PatternLocalExplainer(simplification={
                self.simplification_method}, "
            f"solver={
                self.solver_method}, mode={
                self.mode}, {status})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
