"""
No simplification - identity transformation.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..config.validation import ParameterValidator, validate_array_input
from ..exceptions import ValidationError
from .base import BaseSimplification
from .registry import SimplificationRegistry


@SimplificationRegistry.register("none")
class NoSimplification(BaseSimplification):
    """Identity transformation - no simplification applied."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize NoSimplification.

        Args:
            params: Parameters (unused for identity transformation)
        """
        super().__init__(params)

    @validate_array_input
    def fit(self, X_train: np.ndarray, **kwargs) -> "NoSimplification":
        """Fit the simplification method (no-op for identity).

        Args:
            X_train: Training data, shape (n_samples, n_features)
            **kwargs: Additional arguments (unused)

        Returns:
            Self for method chaining
        """
        ParameterValidator.validate_training_data(X_train)
        self.is_fitted = True
        return self

    @validate_array_input
    def transform_instance(self, instance: np.ndarray) -> np.ndarray:
        """Transform a single instance (identity).

        Args:
            instance: Instance to transform, shape (n_features,)

        Returns:
            Unchanged instance
        """
        if not self.is_fitted:
            raise ValidationError(
                "NoSimplification must be fitted before transforming instances"
            )

        ParameterValidator.validate_array(instance, "instance", ndim=1)
        return instance.copy()

    @validate_array_input
    def transform_training_data(self, X_train: np.ndarray) -> np.ndarray:
        """Transform training data (identity).

        Args:
            X_train: Training data to transform, shape (n_samples, n_features)

        Returns:
            Unchanged training data
        """
        if not self.is_fitted:
            raise ValidationError(
                "NoSimplification must be fitted before transforming data"
            )

        ParameterValidator.validate_array(X_train, "X_train", ndim=2)
        return X_train.copy()

    @validate_array_input
    def inverse_transform_weights(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights back to original space (identity).

        Args:
            weights: Weights in simplified space, shape (n_features,)

        Returns:
            Unchanged weights
        """
        ParameterValidator.validate_array(weights, "weights", ndim=1)
        return weights.copy()

    def create_predict_function(self, original_predict_fn: Callable) -> Callable:
        """Create prediction function for simplified space (identity).

        Args:
            original_predict_fn: Original prediction function

        Returns:
            Same prediction function (no transformation needed)
        """
        if not callable(original_predict_fn):
            raise ValidationError("original_predict_fn must be callable")

        return original_predict_fn

    @property
    def n_features_out(self) -> Optional[int]:
        """Number of output features (same as input for identity)."""
        return None  # Same as input

    @property
    def transformation_type(self) -> str:
        """Type of transformation applied."""
        return "identity"

    def get_feature_names_out(
        self, feature_names_in: Optional[list] = None
    ) -> Optional[list]:
        """Get output feature names.

        Args:
            feature_names_in: Input feature names

        Returns:
            Same feature names (identity transformation)
        """
        return feature_names_in
