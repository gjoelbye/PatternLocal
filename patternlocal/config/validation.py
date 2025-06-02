"""
Parameter validation utilities.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np

from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class PredictFunction(Protocol):
    """Protocol for prediction functions."""

    def __call__(self, X: np.ndarray) -> np.ndarray: ...


def validate_fitted(func: Callable) -> Callable:
    """Decorator to validate that explainer is fitted before method call."""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "is_fitted", False):
            raise ValidationError(f"{func.__name__} requires fitted explainer")
        return func(self, *args, **kwargs)

    return wrapper


def validate_array_input(func: Callable) -> Callable:
    """Decorator to validate numpy array inputs."""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                if arg.size == 0:
                    raise ValidationError(f"Argument {i} is an empty array")
                if np.any(np.isnan(arg)):
                    raise ValidationError(f"Argument {i} contains NaN values")
                if np.any(np.isinf(arg)):
                    raise ValidationError(f"Argument {i} contains infinite values")
        return func(self, *args, **kwargs)

    return wrapper


class ParameterValidator:
    """Utility class for parameter validation."""

    @staticmethod
    def validate_array(
        arr: np.ndarray,
        name: str,
        ndim: Optional[int] = None,
        min_size: int = 1,
        dtype: Optional[type] = None,
    ) -> None:
        """Validate numpy array parameters.

        Args:
            arr: Array to validate
            name: Parameter name for error messages
            ndim: Expected number of dimensions
            min_size: Minimum array size
            dtype: Expected dtype

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(arr, np.ndarray):
            raise ValidationError(f"{name} must be a numpy array")

        if arr.size < min_size:
            raise ValidationError(f"{name} must have at least {min_size} elements")

        if ndim is not None and arr.ndim != ndim:
            raise ValidationError(f"{name} must be {ndim}-dimensional, got {arr.ndim}")

        if dtype is not None and arr.dtype != dtype:
            raise ValidationError(
                f"{name} must have dtype {dtype}, got {
                    arr.dtype}"
            )

        if np.any(np.isnan(arr)):
            raise ValidationError(f"{name} contains NaN values")

        if np.any(np.isinf(arr)):
            raise ValidationError(f"{name} contains infinite values")

    @staticmethod
    def validate_predict_function(
        predict_fn: Callable,
        test_input: np.ndarray,
        expected_output_shape: Optional[tuple] = None,
    ) -> None:
        """Validate prediction function.

        Args:
            predict_fn: Function to validate
            test_input: Test input for function
            expected_output_shape: Expected output shape

        Raises:
            ValidationError: If validation fails
        """
        if not callable(predict_fn):
            raise ValidationError("predict_fn must be callable")

        try:
            # Test with small input
            test_output = predict_fn(test_input)
        except Exception as e:
            raise ValidationError(f"predict_fn failed on test input: {e}")

        if not isinstance(test_output, np.ndarray):
            raise ValidationError("predict_fn must return numpy array")

        if test_output.shape[0] != test_input.shape[0]:
            raise ValidationError(
                f"predict_fn output batch size {test_output.shape[0]} "
                f"doesn't match input batch size {test_input.shape[0]}"
            )

        if expected_output_shape and test_output.shape[1:] != expected_output_shape[1:]:
            raise ValidationError(
                f"predict_fn output shape {test_output.shape} "
                f"doesn't match expected {expected_output_shape}"
            )

        if np.any(np.isnan(test_output)):
            raise ValidationError("predict_fn returns NaN values")

        if np.any(np.isinf(test_output)):
            raise ValidationError("predict_fn returns infinite values")

    @staticmethod
    def validate_lime_params(mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize LIME parameters.

        Args:
            mode: LIME mode ('tabular' or 'image')
            params: LIME parameters

        Returns:
            Validated and normalized parameters

        Raises:
            ValidationError: If validation fails
        """
        validated_params = params.copy()

        # Validate common parameters
        if "num_samples" in validated_params:
            if (
                not isinstance(validated_params["num_samples"], int)
                or validated_params["num_samples"] <= 0
            ):
                raise ValidationError("num_samples must be a positive integer")

        if (
            "kernel_width" in validated_params
            and validated_params["kernel_width"] is not None
        ):
            if (
                not isinstance(validated_params["kernel_width"], (int, float))
                or validated_params["kernel_width"] <= 0
            ):
                raise ValidationError("kernel_width must be a positive number")

        # Mode-specific validation
        if mode == "image":
            if "labels" not in validated_params:
                logger.warning("No labels specified for image mode, using [1]")
                validated_params["labels"] = [1]
            elif not isinstance(validated_params["labels"], list):
                raise ValidationError("labels must be a list for image mode")

        elif mode == "tabular":
            if "discretize_continuous" in validated_params:
                if not isinstance(validated_params["discretize_continuous"], bool):
                    raise ValidationError("discretize_continuous must be boolean")

        return validated_params

    @staticmethod
    def validate_solver_inputs(
        lime_weights: np.ndarray,
        lime_intercept: float,
        instance: np.ndarray,
        X_train: np.ndarray,
    ) -> None:
        """Validate solver input arguments.

        Args:
            lime_weights: LIME explanation weights
            lime_intercept: LIME explanation intercept
            instance: The instance being explained
            X_train: Training data

        Raises:
            ValidationError: If validation fails
        """
        ParameterValidator.validate_array(lime_weights, "lime_weights", ndim=1)
        ParameterValidator.validate_array(instance, "instance", ndim=1)
        ParameterValidator.validate_array(X_train, "X_train", ndim=2)

        if not np.isscalar(lime_intercept):
            raise ValidationError("lime_intercept must be a scalar")

        if np.isnan(lime_intercept) or np.isinf(lime_intercept):
            raise ValidationError("lime_intercept must be finite")

        if lime_weights.shape[0] != instance.shape[0]:
            raise ValidationError("lime_weights and instance must have same length")

        if X_train.shape[1] != instance.shape[0]:
            raise ValidationError(
                "X_train feature dimension must match instance length"
            )

    @staticmethod
    def validate_image_params(
        image_shape: tuple, instance: np.ndarray, simplification_params: Dict[str, Any]
    ) -> None:
        """Validate image-specific parameters.

        Args:
            image_shape: Expected image shape (height, width)
            instance: Image instance (flattened)
            simplification_params: Simplification parameters

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(image_shape, tuple) or len(image_shape) != 2:
            raise ValidationError("image_shape must be a 2-tuple (height, width)")

        if not all(isinstance(dim, int) and dim > 0 for dim in image_shape):
            raise ValidationError("image_shape dimensions must be positive integers")

        expected_size = image_shape[0] * image_shape[1]
        if instance.size != expected_size:
            raise ValidationError(
                f"Instance size {instance.size} doesn't match image_shape {image_shape} "
                f"(expected {expected_size})"
            )

        # Validate superpixel-specific parameters
        if "method" in simplification_params:
            method = simplification_params["method"]
            if method == "slic":
                if "n_segments" in simplification_params:
                    n_segments = simplification_params["n_segments"]
                    if not isinstance(n_segments, int) or n_segments <= 0:
                        raise ValidationError("n_segments must be a positive integer")

                    max_segments = min(image_shape) ** 2
                    if n_segments > max_segments:
                        logger.warning(
                            f"n_segments ({n_segments}) is large for image shape {image_shape}. "
                            f"Consider using <= {max_segments}"
                        )

            elif method == "grid":
                for param in ["grid_rows", "grid_cols"]:
                    if param in simplification_params:
                        value = simplification_params[param]
                        if not isinstance(value, int) or value <= 0:
                            raise ValidationError(f"{param} must be a positive integer")

    @staticmethod
    def validate_training_data(X_train: np.ndarray, min_samples: int = 10) -> None:
        """Validate training data.

        Args:
            X_train: Training data
            min_samples: Minimum number of samples required

        Raises:
            ValidationError: If validation fails
        """
        ParameterValidator.validate_array(X_train, "X_train", ndim=2)

        if X_train.shape[0] < min_samples:
            raise ValidationError(f"X_train must have at least {min_samples} samples")

        # Check for constant features
        feature_vars = np.var(X_train, axis=0)
        constant_features = np.where(feature_vars == 0)[0]
        if len(constant_features) > 0:
            logger.warning(
                f"Found {len(constant_features)} constant features: {constant_features[:10]}"
                f"{'...' if len(constant_features) > 10 else ''}"
            )
