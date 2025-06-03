"""
Tests for kernel functions and registry in PatternLocal.
"""

import numpy as np
import pytest

from patternlocal.exceptions import ValidationError
from patternlocal.utils.kernels import (
    BiweightKernel,
    CosineKernel,
    EpanechnikovKernel,
    GaussianKernel,
    KernelFunction,
    KernelRegistry,
    LogisticKernel,
    TriangularKernel,
    TricubeKernel,
    UniformKernel,
)


class TestKernelFunctions:
    """Test individual kernel function implementations."""

    @pytest.fixture
    def sample_distances(self):
        """Create sample distances for testing."""
        return np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    @pytest.fixture
    def bandwidth(self):
        """Sample bandwidth for testing."""
        return 1.0

    def test_gaussian_kernel(self, sample_distances, bandwidth):
        """Test Gaussian kernel function."""
        kernel = GaussianKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 1.0  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 1)  # Bounded by 1
        assert np.all(np.diff(weights) <= 0)  # Monotonically decreasing

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_epanechnikov_kernel(self, sample_distances, bandwidth):
        """Test Epanechnikov kernel function."""
        kernel = EpanechnikovKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 0.75  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 0.75)  # Bounded by 0.75
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_uniform_kernel(self, sample_distances, bandwidth):
        """Test Uniform kernel function."""
        kernel = UniformKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 1.0  # At distance 0
        assert np.all(np.isin(weights, [0, 1]))  # Binary values
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_triangular_kernel(self, sample_distances, bandwidth):
        """Test Triangular kernel function."""
        kernel = TriangularKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 1.0  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 1)  # Bounded by 1
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_biweight_kernel(self, sample_distances, bandwidth):
        """Test Biweight kernel function."""
        kernel = BiweightKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 15 / 16  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 15 / 16)  # Bounded by 15/16
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_tricube_kernel(self, sample_distances, bandwidth):
        """Test Tricube kernel function."""
        kernel = TricubeKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 70 / 81  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 70 / 81)  # Bounded by 70/81
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_cosine_kernel(self, sample_distances, bandwidth):
        """Test Cosine kernel function."""
        kernel = CosineKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == np.pi / 4  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= np.pi / 4)  # Bounded by pi/4
        assert weights[-1] == 0  # Zero at distance > bandwidth

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)

    def test_logistic_kernel(self, sample_distances, bandwidth):
        """Test Logistic kernel function."""
        kernel = LogisticKernel()
        weights = kernel(sample_distances, bandwidth)

        # Check properties
        assert weights.shape == sample_distances.shape
        assert weights[0] == 0.25  # At distance 0
        assert np.all(weights >= 0)  # Non-negative
        assert np.all(weights <= 0.25)  # Bounded by 0.25
        assert np.all(np.diff(weights) <= 0)  # Monotonically decreasing

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(sample_distances, -1.0)


class TestKernelRegistry:
    """Test the kernel registry functionality."""

    def test_list_available(self):
        """Test listing available kernels."""
        available = KernelRegistry.list_available()
        expected = {
            "gaussian",
            "epanechnikov",
            "uniform",
            "rectangular",
            "triangular",
            "biweight",
            "quartic",
            "tricube",
            "cosine",
            "logistic",
        }
        assert set(available) == expected

    def test_create_kernel(self):
        """Test creating kernel instances."""
        # Test creating each kernel type
        for name in KernelRegistry.list_available():
            kernel = KernelRegistry.create(name)
            assert isinstance(kernel, KernelFunction)

        # Test invalid kernel name
        with pytest.raises(Exception):
            KernelRegistry.create("invalid_kernel")

    def test_is_registered(self):
        """Test kernel registration checking."""
        assert KernelRegistry.is_registered("gaussian")
        assert KernelRegistry.is_registered("epanechnikov")
        assert not KernelRegistry.is_registered("invalid_kernel")

    def test_custom_kernel_registration(self):
        """Test registering a custom kernel function."""

        @KernelRegistry.register("custom")
        class CustomKernel(KernelFunction):
            """Custom kernel for testing registration."""

            def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
                """Simple custom kernel implementation."""
                if bandwidth <= 0:
                    raise ValidationError("bandwidth must be positive")
                return np.exp(-distances / bandwidth)

        # Check registration
        assert KernelRegistry.is_registered("custom")
        assert "custom" in KernelRegistry.list_available()

        # Test created instance
        kernel = KernelRegistry.create("custom")
        assert isinstance(kernel, CustomKernel)

        # Test functionality
        distances = np.array([0.0, 1.0, 2.0])
        weights = kernel(distances, bandwidth=1.0)
        assert weights.shape == distances.shape
        assert weights[0] == 1.0
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

        # Test invalid bandwidth
        with pytest.raises(ValidationError):
            kernel(distances, bandwidth=-1.0)

    def test_kernel_aliases(self):
        """Test that kernel aliases work correctly."""
        # Test rectangular is alias for uniform
        uniform = KernelRegistry.create("uniform")
        rectangular = KernelRegistry.create("rectangular")
        assert isinstance(uniform, UniformKernel)
        assert isinstance(rectangular, UniformKernel)

        # Test quartic is alias for biweight
        biweight = KernelRegistry.create("biweight")
        quartic = KernelRegistry.create("quartic")
        assert isinstance(biweight, BiweightKernel)
        assert isinstance(quartic, BiweightKernel)
