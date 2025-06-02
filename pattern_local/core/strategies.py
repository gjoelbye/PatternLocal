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
    def detect_mode(
        self, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> str:
        """Detect appropriate LIME mode.

        Args:
            simplification: Simplification method
            lime_params: LIME parameters

        Returns:
            LIME mode ('tabular' or 'image')
        """

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


class TabularModeStrategy(LimeModeStrategy):
    """Strategy for tabular data mode."""

    def detect_mode(
        self, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> str:
        """Detect tabular mode."""
        # Check explicit mode specification
        if "mode" in lime_params:
            if lime_params["mode"] == "image":
                logger.warning("Explicit image mode with non-superpixel simplification")
            return lime_params["mode"]

        # Auto-detect: tabular if not superpixel simplification
        if not isinstance(simplification, SuperpixelSimplification):
            return "tabular"

        return "image"  # Superpixel implies image mode

    def create_explainer(
        self,
        X_train: np.ndarray,
        lime_params: Dict[str, Any],
        random_state: Optional[int] = None,
    ) -> lime_tabular.LimeTabularExplainer:
        """Create tabular LIME explainer."""
        validated_params = ParameterValidator.validate_lime_params(
            "tabular", lime_params
        )

        return lime_tabular.LimeTabularExplainer(
            X_train,
            feature_selection=validated_params.get("feature_selection", "auto"),
            discretize_continuous=validated_params.get("discretize_continuous", True),
            kernel_width=validated_params.get("kernel_width", None),
            sample_around_instance=validated_params.get("sample_around_instance", True),
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


class ImageModeStrategy(LimeModeStrategy):
    """Strategy for image data mode."""

    def detect_mode(
        self, simplification: BaseSimplification, lime_params: Dict[str, Any]
    ) -> str:
        """Detect image mode."""
        # Check explicit mode specification
        if "mode" in lime_params and lime_params["mode"] == "image":
            return "image"

        # Auto-detect: image if superpixel simplification
        if isinstance(simplification, SuperpixelSimplification):
            return "image"

        return "tabular"

    def create_explainer(
        self,
        X_train: np.ndarray,
        lime_params: Dict[str, Any],
        random_state: Optional[int] = None,
    ) -> lime_image.LimeImageExplainer:
        """Create image LIME explainer."""
        validated_params = ParameterValidator.validate_lime_params("image", lime_params)

        # Handle kernel_width - image explainer doesn't accept None
        kernel_width = validated_params.get("kernel_width", 0.25)
        if kernel_width is None:
            kernel_width = 0.25

        return lime_image.LimeImageExplainer(
            kernel_width=kernel_width,
            verbose=validated_params.get("verbose", False),
            feature_selection=validated_params.get("feature_selection", "none"),
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
        # Try image strategy first for auto-detection
        image_strategy = cls.create_strategy("image")
        detected_mode = image_strategy.detect_mode(simplification, lime_params)

        return cls.create_strategy(detected_mode)
