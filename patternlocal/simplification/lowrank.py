"""
Low-rank simplification using PCA.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA

from ..config.validation import ParameterValidator, validate_array_input
from ..exceptions import ValidationError
from .base import BaseSimplification
from .registry import SimplificationRegistry


@SimplificationRegistry.register("lowrank")
class LowRankSimplification(BaseSimplification):
    """Low-rank simplification using Principal Component Analysis (PCA).

    This method applies PCA to reduce the dimensionality of the data
    before computing explanations, then projects the results back
    to the original space.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize LowRankSimplification.

        Args:
            params: Parameters for PCA
                - n_components: Number of components to keep (default: 0.95 explained variance)
                - whiten: Whether to whiten the components (default: False)
                - svd_solver: SVD solver to use (default: 'auto')
        """
        super().__init__(params)

        # Set default parameters
        self.n_components = self.params.get("n_components", 0.95)
        self.whiten = self.params.get("whiten", False)
        self.svd_solver = self.params.get("svd_solver", "auto")

        self.pca = None
        self.components_ = None
        self.mean_ = None

    @validate_array_input
    def fit(self, X_train: np.ndarray, **kwargs) -> "LowRankSimplification":
        """Fit PCA to the training data.

        Args:
            X_train: Training data, shape (n_samples, n_features)
            **kwargs: Additional arguments (unused)

        Returns:
            Self for method chaining
        """
        ParameterValidator.validate_training_data(X_train)

        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
        )

        self.pca.fit(X_train)
        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_
        self.is_fitted = True

        return self

    @validate_array_input
    def transform_instance(self, instance: np.ndarray) -> np.ndarray:
        """Transform instance to low-rank representation.

        Args:
            instance: Instance to transform, shape (n_features,)

        Returns:
            Instance in low-rank space
        """
        if not self.is_fitted:
            raise ValidationError(
                "LowRankSimplification must be fitted before transform"
            )

        ParameterValidator.validate_array(instance, "instance", ndim=1)
        return self.pca.transform(instance.reshape(1, -1)).flatten()

    @validate_array_input
    def transform_training_data(self, X_train: np.ndarray) -> np.ndarray:
        """Transform training data to low-rank representation.

        Args:
            X_train: Training data to transform, shape (n_samples, n_features)

        Returns:
            Training data in low-rank space
        """
        if not self.is_fitted:
            raise ValidationError(
                "LowRankSimplification must be fitted before transform"
            )

        ParameterValidator.validate_array(X_train, "X_train", ndim=2)
        return self.pca.transform(X_train)

    @validate_array_input
    def inverse_transform_weights(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights from low-rank space back to original space.

        Args:
            weights: Weights in low-rank space, shape (n_components,)

        Returns:
            Weights in original space
        """
        if not self.is_fitted:
            raise ValidationError(
                "LowRankSimplification must be fitted before inverse_transform"
            )

        ParameterValidator.validate_array(weights, "weights", ndim=1)
        return self.components_.T @ weights

    def create_predict_function(self, original_predict_fn: Callable) -> Callable:
        """Create prediction function that works in low-rank space.

        Args:
            original_predict_fn: Original prediction function

        Returns:
            Prediction function for low-rank space
        """
        if not self.is_fitted:
            raise ValidationError(
                "LowRankSimplification must be fitted before creating predict function"
            )

        if not callable(original_predict_fn):
            raise ValidationError("original_predict_fn must be callable")

        def lowrank_predict_fn(instances_lr):
            """Prediction function for low-rank instances."""
            # Transform back to original space
            instances_original = self.pca.inverse_transform(instances_lr)
            return original_predict_fn(instances_original)

        return lowrank_predict_fn

    @property
    def n_components_(self) -> int:
        """Number of components after fitting."""
        if not self.is_fitted:
            raise ValidationError("LowRankSimplification must be fitted first")
        return self.pca.n_components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Explained variance ratio of each component."""
        if not self.is_fitted:
            raise ValidationError("LowRankSimplification must be fitted first")
        return self.pca.explained_variance_ratio_

    @property
    def transformation_type(self) -> str:
        """Type of transformation applied."""
        return "dimensionality_reduction"

    def get_feature_names_out(self, feature_names_in: Optional[list] = None) -> list:
        """Get output feature names.

        Args:
            feature_names_in: Input feature names

        Returns:
            Principal component names
        """
        if not self.is_fitted:
            return None

        n_components = self.n_components_
        return [f"PC{i + 1}" for i in range(n_components)]
