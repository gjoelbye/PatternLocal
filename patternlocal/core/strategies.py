"""
Strategy pattern implementations for LIME mode detection and explanation generation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from lime import lime_image, lime_tabular

from ..config.validation import ParameterValidator
from ..exceptions import ExplanationError, ValidationError
from ..simplification.base import BaseSimplification
from ..simplification.superpixel import SuperpixelSimplification

logger = logging.getLogger(__name__)


class LimeModeStrategy(ABC):
    """Abstract strategy for LIME mode detection and explanation generation."""

    @abstractmethod
    def create_explainer(
        self,
        X_train: np.ndarray,
        lime_params: Dict[str, Any],
        random_state: Optional[int] = None,
    ) -> Any:
        """Create LIME explainer.

        Args:
            X_train: Training data
            lime_params: LIME parameters
            random_state: Random state

        Returns:
            LIME explainer instance
        """

    @abstractmethod
    def generate_explanation(
        self,
        lime_explainer: Any,
        instance: np.ndarray,
        instance_simplified: np.ndarray,
        predict_fn: Callable,
        simplification: BaseSimplification,
        **kwargs: Any,
    ) -> Any:
        """Generate LIME explanation.

        Args:
            lime_explainer: LIME explainer instance
            instance: Original instance
            instance_simplified: Simplified instance
            predict_fn: Prediction function for simplified space
            simplification: Simplification method
            **kwargs: Additional arguments

        Returns:
            LIME explanation object
        """

    @abstractmethod
    def extract_explanation(self, explanation: Any, num_features: int) -> tuple:
        """Extract weights and intercept from LIME explanation.

        Args:
            explanation: LIME explanation object
            num_features: Number of features

        Returns:
            Tuple of (weights, intercept)
        """


# ToDo: Add feature selection


class TabularModeStrategy(LimeModeStrategy):
    """Strategy for tabular data mode."""

    # Set of known parameters for tabular mode
    KNOWN_PARAMS = {
        "num_samples",
        "bandwidth",
        "discretize_continuous",
        # "feature_selection",  # TODO: Implement feature selection
        "sample_around_instance",
        "mode",  # Common parameter for mode selection
    }

    @staticmethod
    def validate_lime_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize tabular LIME parameters.

        Args:
            params: LIME parameters

        Returns:
            Validated and normalized parameters

        Raises:
            ValidationError: If validation fails or unknown parameters are provided
        """
        validated_params = params.copy()

        # Check for unknown parameters
        unknown_params = set(validated_params.keys()) - TabularModeStrategy.KNOWN_PARAMS
        if unknown_params:
            raise ValidationError(
                f"Unknown parameters for tabular mode: {unknown_params}. "
                f"Known parameters are: {sorted(TabularModeStrategy.KNOWN_PARAMS)}"
            )

        # Validate common parameters
        if "num_samples" in validated_params:
            if (
                not isinstance(validated_params["num_samples"], int)
                or validated_params["num_samples"] <= 0
            ):
                raise ValidationError("num_samples must be a positive integer")

        if (
            "bandwidth" in validated_params
            and validated_params["bandwidth"] is not None
        ):
            if (
                not isinstance(validated_params["bandwidth"], (int, float))
                or validated_params["bandwidth"] <= 0
            ):
                raise ValidationError("bandwidth must be a positive number")

        # Tabular-specific validation
        if "discretize_continuous" in validated_params:
            if not isinstance(validated_params["discretize_continuous"], bool):
                raise ValidationError("discretize_continuous must be boolean")

        # TODO: Implement feature selection validation
        # if "feature_selection" in validated_params:
        #     valid_selections = ["none", "auto", "forward", "lasso_path", "highest_weights"]
        #     if validated_params["feature_selection"] not in valid_selections:
        #         raise ValidationError(f"feature_selection must be one of {valid_selections}")

        if "sample_around_instance" in validated_params:
            if not isinstance(validated_params["sample_around_instance"], bool):
                raise ValidationError("sample_around_instance must be boolean")

        # Force feature selection to none until implemented
        validated_params["feature_selection"] = "none"

        return validated_params

    def create_explainer(
        self,
        X_train: np.ndarray,
        lime_params: Dict[str, Any],
        random_state: Optional[int] = None,
    ) -> lime_tabular.LimeTabularExplainer:
        """Create tabular LIME explainer."""
        validated_params = self.validate_lime_params(lime_params)

        return lime_tabular.LimeTabularExplainer(
            X_train,
            feature_selection="none",  # Force feature selection to none until implemented
            discretize_continuous=validated_params.get("discretize_continuous", False),
            kernel_width=validated_params.get("bandwidth", None),
            sample_around_instance=validated_params.get(
                "sample_around_instance", False
            ),
            random_state=random_state,
        )

    def generate_explanation(
        self,
        lime_explainer: lime_tabular.LimeTabularExplainer,
        instance: np.ndarray,
        instance_simplified: np.ndarray,
        predict_fn: Callable,
        simplification: BaseSimplification,
        **kwargs,
    ) -> Any:
        """Generate tabular explanation."""
        num_samples = kwargs.get("num_samples", 5000)
        num_features = kwargs.get("num_features", instance_simplified.shape[0])

        # Validate prediction function
        test_input = instance_simplified.reshape(1, -1)
        ParameterValidator.validate_predict_function(predict_fn, test_input)

        return lime_explainer.explain_instance(
            instance_simplified,
            predict_fn,
            num_samples=num_samples,
            num_features=num_features,
        )

    def extract_explanation(
        self, explanation: Any, num_features: int
    ) -> Tuple[np.ndarray, float]:
        """Extract weights and intercept from tabular explanation."""
        # Get the explanation for the first class (assuming binary/regression)
        local_exp = explanation.local_exp[list(explanation.local_exp.keys())[0]]

        # Initialize weights array
        weights = np.zeros(num_features)

        # Fill in the weights from LIME explanation
        for feature_idx, weight in local_exp:
            if 0 <= feature_idx < num_features:
                weights[feature_idx] = weight

        # Get intercept
        intercept = explanation.intercept[list(explanation.intercept.keys())[0]]

        return weights, intercept

    def detect_mode(
        self, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> str:
        """Detect LIME mode for tabular data."""
        return "tabular"


class ImageModeStrategy(LimeModeStrategy):
    """Strategy for image data mode."""

    # Set of known parameters for image mode
    KNOWN_PARAMS = {
        "num_samples",
        "bandwidth",
        "labels",
        # "feature_selection",  # TODO: Implement feature selection
        "verbose",
        "hide_color",
        "mode",  # Common parameter for mode selection
    }

    @staticmethod
    def validate_lime_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize image LIME parameters.

        Args:
            params: LIME parameters

        Returns:
            Validated and normalized parameters

        Raises:
            ValidationError: If validation fails or unknown parameters are provided
        """
        validated_params = params.copy()

        # Check for unknown parameters
        unknown_params = set(validated_params.keys()) - ImageModeStrategy.KNOWN_PARAMS
        if unknown_params:
            raise ValidationError(
                f"Unknown parameters for image mode: {unknown_params}. "
                f"Known parameters are: {sorted(ImageModeStrategy.KNOWN_PARAMS)}"
            )

        # Validate common parameters
        if "num_samples" in validated_params:
            if (
                not isinstance(validated_params["num_samples"], int)
                or validated_params["num_samples"] <= 0
            ):
                raise ValidationError("num_samples must be a positive integer")

        if (
            "bandwidth" in validated_params
            and validated_params["bandwidth"] is not None
        ):
            if (
                not isinstance(validated_params["bandwidth"], (int, float))
                or validated_params["bandwidth"] <= 0
            ):
                raise ValidationError("bandwidth must be a positive number")

        # Image-specific validation
        if "labels" not in validated_params:
            logger.warning("No labels specified for image mode, using [1]")
            validated_params["labels"] = [1]
        elif not isinstance(validated_params["labels"], list):
            raise ValidationError("labels must be a list for image mode")

        # TODO: Implement feature selection validation
        # if "feature_selection" in validated_params:
        #     valid_selections = ["none", "auto", "forward", "lasso_path", "highest_weights"]
        #     if validated_params["feature_selection"] not in valid_selections:
        #         raise ValidationError(f"feature_selection must be one of {valid_selections}")

        if "verbose" in validated_params:
            if not isinstance(validated_params["verbose"], bool):
                raise ValidationError("verbose must be boolean")

        if "hide_color" in validated_params:
            if not isinstance(validated_params["hide_color"], (int, float)):
                raise ValidationError("hide_color must be a number")

        # Force feature selection to none until implemented
        validated_params["feature_selection"] = "none"

        return validated_params

    def create_explainer(
        self,
        X_train: np.ndarray,
        lime_params: Dict[str, Any],
        random_state: Optional[int] = None,
    ) -> lime_image.LimeImageExplainer:
        """Create image LIME explainer."""
        validated_params = self.validate_lime_params(lime_params)

        # Handle bandwidth - image explainer doesn't accept None
        bandwidth = validated_params.get("bandwidth", 0.25)
        if bandwidth is None:
            bandwidth = 0.25

        return lime_image.LimeImageExplainer(
            kernel_width=bandwidth,
            verbose=validated_params.get("verbose", False),
            feature_selection="none",  # Force feature selection to none until implemented
            random_state=random_state,
        )

    def generate_explanation(
        self,
        lime_explainer: lime_image.LimeImageExplainer,
        instance: np.ndarray,
        instance_simplified: np.ndarray,
        predict_fn: Callable,
        simplification: BaseSimplification,
        **kwargs: Any,
    ) -> Any:
        """Generate image explanation."""
        # Get image shape from simplification
        if (
            not hasattr(simplification, "image_shape")
            or simplification.image_shape is None
        ):
            raise ExplanationError(
                "SuperpixelSimplification must have image_shape defined for image mode"
            )

        image_shape = simplification.image_shape

        # Reshape instance to image format
        image_2d = instance.reshape(image_shape)

        # Add channel dimension for LIME (expects H x W x C format)
        if len(image_2d.shape) == 2:
            image_3d = np.expand_dims(image_2d, axis=2)  # Add channel dimension
        else:
            image_3d = image_2d

        # Create segmentation function that returns 2D segments
        def segmentation_fn(img):
            # LIME expects segmentation function to return 2D array
            # Our SuperpixelSimplification.segments is already flattened, so
            # reshape it
            return simplification.segments.reshape(image_shape)

        # Create prediction function for LIME image format
        def predict_fn_image(images):
            """Prediction function that handles LIME's image format."""
            # LIME passes images with shape (batch_size, height, width,
            # channels)
            batch_size = images.shape[0]
            # Flatten to get back to our expected format
            images_flat = images.reshape(batch_size, -1)

            # Remove channel dimension if it was added
            if images.shape[-1] == 1:
                images_flat = images_flat[:, : image_shape[0] * image_shape[1]]

            # Get predictions from the simplified prediction function
            predictions = predict_fn(images_flat)

            # Ensure predictions are in the right format for LIME
            # LIME expects a 2D array with shape (batch_size, n_classes)
            if len(predictions.shape) == 1:
                # If 1D, assume binary classification and create 2-column
                # matrix
                predictions_2d = np.column_stack([1 - predictions, predictions])
                return predictions_2d
            else:
                return predictions

        # Get parameters
        labels = kwargs.get(
            "labels", [1]
        )  # Default to class 1 for binary classification
        num_samples = kwargs.get("num_samples", 1000)
        num_features = kwargs.get("num_features", simplification.n_superpixels)
        hide_color = kwargs.get("hide_color", 0)

        # Generate explanation
        return lime_explainer.explain_instance(
            image_3d,
            predict_fn_image,
            labels=labels,
            num_features=num_features,
            num_samples=num_samples,
            hide_color=hide_color,
            segmentation_fn=segmentation_fn,
            random_seed=kwargs.get("random_state"),
        )

    def extract_explanation(
        self, explanation: Any, num_features: int
    ) -> Tuple[np.ndarray, float]:
        """Extract weights and intercept from image explanation."""
        # Image explanations structure: explanation.local_exp[label] contains
        # (segment_id, weight) pairs
        label_key = list(explanation.local_exp.keys())[0]
        local_exp = explanation.local_exp[label_key]

        # Initialize weights array for segments
        weights = np.zeros(num_features)

        # Fill in the weights from LIME explanation
        for segment_id, weight in local_exp:
            if 0 <= segment_id < num_features:
                weights[segment_id] = weight

        # Get intercept
        intercept = explanation.intercept[label_key]

        return weights, intercept

    def detect_mode(
        self, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> str:
        """Detect LIME mode for image data."""
        return "image"


class StrategyFactory:
    """Factory for creating strategy instances."""

    _strategies: Dict[str, type[LimeModeStrategy]] = {
        "tabular": TabularModeStrategy,
        "image": ImageModeStrategy,
    }

    @classmethod
    def create_strategy(cls, mode: str) -> LimeModeStrategy:
        """Create strategy for given mode.

        Args:
            mode: LIME mode ('tabular' or 'image')

        Returns:
            Strategy instance

        Raises:
            ValidationError: If mode is not supported
        """
        if mode not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValidationError(
                f"Unsupported LIME mode: {mode}. Available: {available}"
            )

        return cls._strategies[mode]()

    @classmethod
    def auto_detect_strategy(
        cls, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> LimeModeStrategy:
        """Auto-detect and create appropriate strategy.

        Args:
            simplification: Simplification method
            lime_params: LIME parameters

        Returns:
            Strategy instance
        """

        if "mode" in lime_params:
            if lime_params["mode"] in StrategyFactory._strategies:
                return cls.create_strategy(lime_params["mode"])

        elif isinstance(simplification, SuperpixelSimplification):
            return cls.create_strategy("image")

        else:
            return cls.create_strategy("tabular")
