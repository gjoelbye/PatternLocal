"""
Base class for simplification methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np


class BaseSimplification(ABC):
    """Abstract base class for data simplification methods.

    Simplification methods transform the data into a different representation
    before applying LIME and pattern computation. Examples include dimensionality
    reduction (PCA) or image segmentation (superpixels).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the simplification method.

        Args:
            params: Method-specific parameters
        """
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, **kwargs) -> "BaseSimplification":
        """Fit the simplification method to training data.

        Args:
            X_train: Training data
            **kwargs: Additional method-specific arguments

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def transform_instance(self, instance: np.ndarray) -> np.ndarray:
        """Transform a single instance to simplified representation.

        Args:
            instance: Instance to transform

        Returns:
            Transformed instance
        """

    @abstractmethod
    def transform_training_data(self, X_train: np.ndarray) -> np.ndarray:
        """Transform training data to simplified representation.

        Args:
            X_train: Training data to transform

        Returns:
            Transformed training data
        """

    @abstractmethod
    def inverse_transform_weights(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights from simplified space back to original space.

        Args:
            weights: Weights in simplified space

        Returns:
            Weights in original space
        """

    @abstractmethod
    def create_predict_function(self, original_predict_fn: Callable) -> Callable:
        """Create a prediction function that works in the simplified space.

        Args:
            original_predict_fn: Original prediction function

        Returns:
            Prediction function for simplified space
        """

    def fit_transform_instance(
        self, instance: np.ndarray, X_train: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Convenience method to fit and transform an instance.

        Args:
            instance: Instance to transform
            X_train: Training data to fit on
            **kwargs: Additional method-specific arguments

        Returns:
            Transformed instance
        """
        if not self.is_fitted:
            self.fit(X_train, **kwargs)
        return self.transform_instance(instance)

    def fit_transform_training_data(self, X_train: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience method to fit and transform training data.

        Args:
            X_train: Training data
            **kwargs: Additional method-specific arguments

        Returns:
            Transformed training data
        """
        if not self.is_fitted:
            self.fit(X_train, **kwargs)
        return self.transform_training_data(X_train)
