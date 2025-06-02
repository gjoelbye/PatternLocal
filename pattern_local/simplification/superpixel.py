"""
Superpixel simplification for image data.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base import BaseSimplification
from .registry import SimplificationRegistry


@SimplificationRegistry.register("superpixel")
class SuperpixelSimplification(BaseSimplification):
    """Superpixel simplification for image data.

    This method segments images into superpixels and works with segment-level
    features instead of individual pixels. This reduces dimensionality
    significantly for image data while preserving spatial structure.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize SuperpixelSimplification.

        Args:
            params: Parameters for superpixel segmentation
                - segmentation_fn: Custom segmentation function (default: None)
                - image_shape: Shape of images as (height, width)
                - method: Segmentation method 'slic' or 'grid' (default: 'slic')
                - n_segments: Number of segments for SLIC (default: 200)
                - compactness: SLIC compactness parameter (default: 8)
                - sigma: SLIC sigma parameter (default: 0)
                - grid_rows: Number of grid rows for grid method (default: 8)
                - grid_cols: Number of grid columns for grid method (default: 8)
        """
        super().__init__(params)

        # Segmentation parameters
        self.segmentation_fn = self.params.get("segmentation_fn", None)
        self.image_shape = self.params.get("image_shape", None)
        self.method = self.params.get("method", "slic")

        # SLIC parameters
        self.n_segments = self.params.get("n_segments", 200)
        self.compactness = self.params.get("compactness", 8)
        self.sigma = self.params.get("sigma", 0)

        # Grid parameters
        self.grid_rows = self.params.get("grid_rows", 8)
        self.grid_cols = self.params.get("grid_cols", 8)

        # Internal state
        self.segments_ = None
        self.n_superpixels_ = None

    def fit(self, X_train: np.ndarray, **kwargs) -> "SuperpixelSimplification":
        """Fit the superpixel segmentation.

        Args:
            X_train: Training data (images as flattened arrays)
            **kwargs: Additional arguments
                - image_shape: Override image shape if not provided in params

        Returns:
            Self for method chaining
        """
        # Get image shape
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]

        if self.image_shape is None:
            raise ValueError(
                "image_shape must be provided either in params or as kwarg"
            )

        # Use first training image to create segmentation
        first_image = X_train[0] if len(X_train.shape) > 1 else X_train

        if self.segmentation_fn is not None:
            # Use custom segmentation function
            self.segments_ = self.segmentation_fn(first_image)
        else:
            # Use built-in segmentation methods
            self.segments_ = self._create_segmentation(first_image)

        self.n_superpixels_ = len(np.unique(self.segments_))
        self.is_fitted = True

        return self

    def _create_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Create segmentation using built-in methods."""
        if self.method == "slic":
            return self._slic_segmentation(image)
        elif self.method == "grid":
            return self._grid_segmentation()
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

    def _slic_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply SLIC segmentation."""
        try:
            from skimage.color import gray2rgb
            from skimage.segmentation import slic
        except ImportError:
            raise ImportError("scikit-image is required for SLIC segmentation")

        # Reshape image to 2D
        image_2d = image.reshape(self.image_shape)

        # Convert to RGB if needed
        if len(image_2d.shape) == 2:
            image_rgb = gray2rgb(image_2d)
        else:
            image_rgb = image_2d

        # Apply SLIC
        segments = slic(
            image_rgb,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0,
        )

        return segments.flatten()

    def _grid_segmentation(self) -> np.ndarray:
        """Apply grid segmentation."""
        h, w = self.image_shape

        if h % self.grid_rows != 0:
            raise ValueError(
                f"Image height {h} not divisible by grid_rows {self.grid_rows}"
            )
        if w % self.grid_cols != 0:
            raise ValueError(
                f"Image width {w} not divisible by grid_cols {self.grid_cols}"
            )

        cell_height = h // self.grid_rows
        cell_width = w // self.grid_cols
        seg_map = np.zeros((h, w), dtype=np.int32)

        label = 0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y_start = i * cell_height
                y_end = (i + 1) * cell_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width
                seg_map[y_start:y_end, x_start:x_end] = label
                label += 1

        return seg_map.flatten()

    def transform_instance(self, instance: np.ndarray) -> np.ndarray:
        """Transform instance to superpixel representation.

        Args:
            instance: Image instance as flattened array

        Returns:
            Instance in superpixel space (segment means)
        """
        if not self.is_fitted:
            raise ValueError("SuperpixelSimplification must be fitted before transform")

        return self._extract_segment_means(instance.reshape(1, -1))[0]

    def transform_training_data(self, X_train: np.ndarray) -> np.ndarray:
        """Transform training data to superpixel representation.

        Args:
            X_train: Training data (images as flattened arrays)

        Returns:
            Training data in superpixel space
        """
        if not self.is_fitted:
            raise ValueError("SuperpixelSimplification must be fitted before transform")

        return self._extract_segment_means(X_train)

    def _extract_segment_means(self, X: np.ndarray) -> np.ndarray:
        """Extract mean values for each segment from images."""
        unique_segments, seg_index = np.unique(self.segments_, return_inverse=True)
        num_segments = len(unique_segments)

        # Create mapping matrix
        counts = np.bincount(seg_index)
        M = np.zeros((len(self.segments_), num_segments))
        M[np.arange(len(self.segments_)), seg_index] = 1

        # Compute segment means
        segment_sums = X.dot(M)
        return segment_sums / counts

    def inverse_transform_weights(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights from superpixel space back to pixel space.

        Args:
            weights: Weights in superpixel space

        Returns:
            Weights mapped back to pixel space
        """
        if not self.is_fitted:
            raise ValueError(
                "SuperpixelSimplification must be fitted before inverse_transform"
            )

        # Map segment weights back to pixels
        pixel_weights = np.zeros(len(self.segments_))
        unique_segments = np.unique(self.segments_)

        for i, segment_id in enumerate(unique_segments):
            pixel_weights[self.segments_ == segment_id] = weights[i]

        return pixel_weights

    def create_predict_function(self, original_predict_fn: Callable) -> Callable:
        """Create prediction function that works in superpixel space.

        Args:
            original_predict_fn: Original prediction function

        Returns:
            Prediction function for superpixel space
        """
        if not self.is_fitted:
            raise ValueError(
                "SuperpixelSimplification must be fitted before creating predict function"
            )

        def superpixel_predict_fn(instances_sp):
            """Prediction function for superpixel instances."""
            # Convert superpixel instances back to pixel space
            if len(instances_sp.shape) == 1:
                instances_sp = instances_sp.reshape(1, -1)

            instances_pixel = np.zeros((instances_sp.shape[0], len(self.segments_)))
            unique_segments = np.unique(self.segments_)

            for i, segment_id in enumerate(unique_segments):
                mask = self.segments_ == segment_id
                instances_pixel[:, mask] = instances_sp[:, i : i + 1]

            return original_predict_fn(instances_pixel)

        return superpixel_predict_fn

    @property
    def n_superpixels(self) -> int:
        """Number of superpixels after fitting."""
        if not self.is_fitted:
            raise ValueError("SuperpixelSimplification must be fitted first")
        return self.n_superpixels_

    @property
    def segments(self) -> np.ndarray:
        """Segmentation map."""
        if not self.is_fitted:
            raise ValueError("SuperpixelSimplification must be fitted first")
        return self.segments_
