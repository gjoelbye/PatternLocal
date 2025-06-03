"""
Tests for superpixel functionality and conversions.
"""

import numpy as np
import pytest
from skimage.color import gray2rgb

from patternlocal.simplification import SuperpixelSimplification
from patternlocal.utils import superpixel


@pytest.fixture
def sample_image():
    """Create a sample 8x8 grayscale test image."""
    image = np.zeros((8, 8))
    # Add some patterns to make segmentation meaningful
    image[2:6, 2:6] = 1.0  # Center square
    image[0:2, 0:2] = 0.5  # Top-left square
    return image


@pytest.fixture
def sample_image_rgb(sample_image):
    """Convert sample image to RGB format."""
    return gray2rgb(sample_image)


def test_grid_segmentation_basic(sample_image):
    """Test basic grid segmentation functionality."""
    # Test 2x2 grid
    segments = superpixel.grid_segmentation(
        sample_image.flatten(), dims=(8, 8), grid_rows=2, grid_cols=2
    )

    # Reshape for easier verification
    segments_2d = segments.reshape(8, 8)

    # Check segment count and values
    assert len(np.unique(segments)) == 4
    assert segments_2d[0:4, 0:4].flatten().tolist() == [0] * 16  # Top-left segment
    assert segments_2d[0:4, 4:8].flatten().tolist() == [1] * 16  # Top-right segment
    assert segments_2d[4:8, 0:4].flatten().tolist() == [2] * 16  # Bottom-left segment
    assert segments_2d[4:8, 4:8].flatten().tolist() == [3] * 16  # Bottom-right segment


def test_grid_segmentation_validation(sample_image):
    """Test grid segmentation input validation."""
    # Test invalid grid dimensions
    with pytest.raises(AssertionError):
        superpixel.grid_segmentation(
            sample_image.flatten(), dims=(8, 8), grid_rows=3, grid_cols=2
        )

    with pytest.raises(AssertionError):
        superpixel.grid_segmentation(
            sample_image.flatten(), dims=(8, 8), grid_rows=2, grid_cols=3
        )


def test_slic_segmentation_basic(sample_image):
    """Test basic SLIC segmentation functionality."""
    segments = superpixel.slic_segmentation(
        sample_image.flatten(), dims=(8, 8), n_segments=4
    )

    # Basic checks
    assert len(segments) == 64  # Flattened 8x8 image
    assert len(np.unique(segments)) <= 4  # Should have at most 4 segments
    assert segments.dtype == np.int32  # Now enforced by implementation


def test_superpixel_simplification_grid(sample_image):
    """Test SuperpixelSimplification with grid method."""
    simplification = SuperpixelSimplification(
        {"image_shape": (8, 8), "method": "grid", "grid_rows": 2, "grid_cols": 2}
    )

    # Fit and transform
    X_train = np.array([sample_image.flatten()])
    simplification.fit(X_train)

    # Transform instance
    instance = sample_image.flatten()
    transformed = simplification.transform_instance(instance)

    # Should have 4 features (one per grid cell)
    assert transformed.shape == (4,)

    # Test inverse transform
    weights = np.array([1.0, -1.0, 0.5, -0.5])  # Example weights
    inverse = simplification.inverse_transform_weights(weights)

    assert inverse.shape == (64,)  # Back to original 8x8 flattened shape
    assert len(np.unique(inverse)) == 4  # Should have 4 unique values


def test_superpixel_simplification_slic(sample_image):
    """Test SuperpixelSimplification with SLIC method."""
    simplification = SuperpixelSimplification(
        {
            "image_shape": (8, 8),
            "method": "slic",
            "n_segments": 4,
            "compactness": 8,
            "sigma": 0,
        }
    )

    # Fit and transform
    X_train = np.array([sample_image.flatten()])
    simplification.fit(X_train)

    # Transform instance
    instance = sample_image.flatten()
    transformed = simplification.transform_instance(instance)

    # Should have at most 4 features
    assert transformed.shape[0] <= 4

    # Test inverse transform
    weights = np.ones(transformed.shape[0])
    inverse = simplification.inverse_transform_weights(weights)

    assert inverse.shape == (64,)  # Back to original 8x8 flattened shape


def test_prediction_function_conversion(sample_image):
    """Test prediction function conversion in superpixel space."""
    simplification = SuperpixelSimplification(
        {"image_shape": (8, 8), "method": "grid", "grid_rows": 2, "grid_cols": 2}
    )

    # Mock prediction function that returns binary classification probabilities
    def original_predict_fn(X):
        # Simple threshold-based classifier
        return (X.mean(axis=1) > 0.5).astype(float)

    # Fit simplification
    X_train = np.array([sample_image.flatten()])
    simplification.fit(X_train)

    # Create converted prediction function
    predict_fn_simplified = simplification.create_predict_function(original_predict_fn)

    # Test prediction on transformed instance
    instance = sample_image.flatten()
    instance_simplified = simplification.transform_instance(instance)

    prediction = predict_fn_simplified(instance_simplified.reshape(1, -1))
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)


def test_conversion_consistency(sample_image):
    """Test consistency between utils.superpixel and SuperpixelSimplification."""
    # Create segments using utils.superpixel
    utils_segments = superpixel.grid_segmentation(
        sample_image.flatten(), dims=(8, 8), grid_rows=2, grid_cols=2
    )

    # Create segments using SuperpixelSimplification
    simplification = SuperpixelSimplification(
        {"image_shape": (8, 8), "method": "grid", "grid_rows": 2, "grid_cols": 2}
    )
    X_train = np.array([sample_image.flatten()])
    simplification.fit(X_train)

    # Compare segment counts and structure
    assert len(np.unique(utils_segments)) == len(np.unique(simplification.segments))
    assert utils_segments.shape == simplification.segments.shape


def test_rgb_handling(sample_image_rgb):
    """Test handling of RGB images."""
    # Get flattened RGB image
    rgb_flat = sample_image_rgb.reshape(-1)

    # Test SLIC segmentation with RGB
    segments_rgb = superpixel.slic_segmentation(rgb_flat, dims=(8, 8), n_segments=4)

    # Basic checks for RGB segmentation
    assert len(segments_rgb) == 64  # Should be H*W, not H*W*C
    assert segments_rgb.dtype == np.int32
    assert len(np.unique(segments_rgb)) <= 4

    # Test SuperpixelSimplification with RGB
    simplification = SuperpixelSimplification(
        {"image_shape": (8, 8), "method": "slic", "n_segments": 4}
    )

    # Fit with RGB data
    X_train = np.array([rgb_flat])
    simplification.fit(X_train)

    # Transform should work with RGB data
    transformed_rgb = simplification.transform_instance(rgb_flat)

    # For RGB, each segment has 3 values (R,G,B)
    n_segments = len(transformed_rgb) // 3
    assert n_segments <= 4  # Should have at most 4 segments

    # Test that the values make sense
    transformed_reshaped = transformed_rgb.reshape(-1, 3)
    assert transformed_reshaped.shape[1] == 3  # Each segment has RGB values
    assert np.all(
        (transformed_reshaped >= 0) & (transformed_reshaped <= 1)
    )  # Valid RGB values
