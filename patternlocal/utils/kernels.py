"""
Kernel functions for local weighting in PatternLocal explanations.

This module provides a collection of kernel functions for computing local weights
in PatternLocal explanations. The kernels are used to weight training instances
based on their distance to the instance being explained.

The module includes:
1. A registry system for managing kernel functions
2. Built-in kernel implementations (Gaussian, Epanechnikov, etc.)
3. Support for custom kernel functions through registration

Custom kernels can be registered using the @KernelRegistry.register decorator.
Example:
    @KernelRegistry.register("my_kernel")
    class MyKernel(KernelFunction):
        def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
            # Custom kernel implementation
            return np.exp(-distances / bandwidth)

Available built-in kernels:
- gaussian: Gaussian (RBF) kernel
- epanechnikov: Epanechnikov kernel
- uniform/rectangular: Uniform (box) kernel
- triangular: Triangular kernel
- biweight/quartic: Biweight (quartic) kernel
- tricube: Tricube kernel
- cosine: Cosine kernel
- logistic: Logistic kernel
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import ValidationError
from ..utils.registry import BaseRegistry


class KernelFunction(ABC):
    """Abstract base class for kernel functions in PatternLocal explanations.

    All kernel functions must inherit from this class and implement the __call__
    method. The kernel function computes weights for instances based on their
    distance to the instance being explained.

    Example:
        @KernelRegistry.register("custom")
        class CustomKernel(KernelFunction):
            def __init__(self, params: Optional[Dict[str, Any]] = None):
                super().__init__()
                self.params = params or {}

            def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
                if bandwidth <= 0:
                    raise ValidationError("bandwidth must be positive")
                return np.exp(-distances / bandwidth)

    Attributes:
        params: Optional dictionary of parameters for the kernel

    Methods:
        __call__(distances, bandwidth): Compute kernel weights for distances
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize kernel function.

        Args:
            params: Optional dictionary of parameters for the kernel
        """
        self.params = params or {}

    @abstractmethod
    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute kernel weights for distances.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """


class GaussianKernel(KernelFunction):
    """Gaussian (RBF) kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Gaussian kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute Gaussian kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for Gaussian kernel")
        return np.exp(-0.5 * (distances / bandwidth) ** 2)


class EpanechnikovKernel(KernelFunction):
    """Epanechnikov kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Epanechnikov kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute Epanechnikov kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for Epanechnikov kernel")
        z = distances / bandwidth
        weights = np.zeros_like(distances)
        mask = z <= 1
        weights[mask] = 0.75 * (1 - z[mask] ** 2)
        return weights


class UniformKernel(KernelFunction):
    """Uniform (box) kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Uniform kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute uniform kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for uniform kernel")
        return (distances <= bandwidth).astype(float)


class TriangularKernel(KernelFunction):
    """Triangular kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Triangular kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute triangular kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for triangular kernel")
        z = distances / bandwidth
        weights = np.zeros_like(distances)
        mask = z <= 1
        weights[mask] = 1 - z[mask]
        return weights


class BiweightKernel(KernelFunction):
    """Biweight (quartic) kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Biweight kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute biweight kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for biweight kernel")
        z = distances / bandwidth
        weights = np.zeros_like(distances)
        mask = z <= 1
        weights[mask] = (15 / 16) * (1 - z[mask] ** 2) ** 2
        return weights


class TricubeKernel(KernelFunction):
    """Tricube kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Tricube kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute tricube kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for tricube kernel")
        z = distances / bandwidth
        weights = np.zeros_like(distances)
        mask = z <= 1
        weights[mask] = (70 / 81) * (1 - z[mask] ** 3) ** 3
        return weights


class CosineKernel(KernelFunction):
    """Cosine kernel function."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Cosine kernel.

        Args:
            params: Optional dictionary of parameters (not used for this kernel)
        """
        super().__init__(params)

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute cosine kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for cosine kernel")
        z = distances / bandwidth
        weights = np.zeros_like(distances)
        mask = z <= 1
        weights[mask] = (np.pi / 4) * np.cos(np.pi / 2 * z[mask])
        return weights


class LogisticKernel(KernelFunction):
    """Logistic kernel function."""

    def __call__(self, distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute logistic kernel weights.

        Args:
            distances: Array of distances
            bandwidth: Kernel bandwidth parameter

        Returns:
            Array of kernel weights
        """
        if bandwidth <= 0:
            raise ValidationError("bandwidth must be positive for logistic kernel")
        z = distances / bandwidth
        return 1 / (np.exp(z) + 2 + np.exp(-z))


# Create singleton registry instance
_registry = BaseRegistry(KernelFunction, "kernel")


class KernelRegistry:
    """Registry for managing kernel functions in PatternLocal explanations.

    This class provides a centralized registry for kernel functions, allowing
    easy registration and instantiation of both built-in and custom kernels.

    The registry supports:
    1. Registration of custom kernels using the @register decorator
    2. Creation of kernel instances using create()
    3. Listing available kernels using list_available()
    4. Checking kernel registration using is_registered()

    Example:
        # Register a custom kernel
        @KernelRegistry.register("my_kernel")
        class MyKernel(KernelFunction):
            def __call__(self, distances, bandwidth):
                return np.exp(-distances / bandwidth)

        # Create and use a kernel
        kernel = KernelRegistry.create("my_kernel")
        weights = kernel(distances, bandwidth=1.0)

    Class methods:
        register(name): Decorator to register a kernel function
        create(name, params): Create a kernel function instance
        list_available(): List all available kernel functions
        is_registered(name): Check if a kernel function is registered
    """

    @classmethod
    def register(cls, name: str):
        """Decorator to register a kernel function."""
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, params: Optional[Dict[str, Any]] = None):
        """Create a kernel function instance."""
        return _registry.create(name, params)

    @classmethod
    def list_available(cls):
        """List all available kernel functions."""
        return _registry.list_available()

    @classmethod
    def is_registered(cls, name: str):
        """Check if a kernel function is registered."""
        return _registry.is_registered(name)


# Register built-in kernels
@KernelRegistry.register("gaussian")
class RegisteredGaussianKernel(GaussianKernel):
    """Registered Gaussian kernel function."""

    pass


@KernelRegistry.register("epanechnikov")
class RegisteredEpanechnikovKernel(EpanechnikovKernel):
    """Registered Epanechnikov kernel function."""

    pass


@KernelRegistry.register("uniform")
class RegisteredUniformKernel(UniformKernel):
    """Registered uniform kernel function."""

    pass


@KernelRegistry.register("rectangular")
class RegisteredRectangularKernel(UniformKernel):
    """Registered rectangular kernel function (alias for uniform)."""

    pass


@KernelRegistry.register("triangular")
class RegisteredTriangularKernel(TriangularKernel):
    """Registered triangular kernel function."""

    pass


@KernelRegistry.register("biweight")
class RegisteredBiweightKernel(BiweightKernel):
    """Registered biweight kernel function."""

    pass


@KernelRegistry.register("quartic")
class RegisteredQuarticKernel(BiweightKernel):
    """Registered quartic kernel function (alias for biweight)."""

    pass


@KernelRegistry.register("tricube")
class RegisteredTricubeKernel(TricubeKernel):
    """Registered tricube kernel function."""

    pass


@KernelRegistry.register("cosine")
class RegisteredCosineKernel(CosineKernel):
    """Registered cosine kernel function."""

    pass


@KernelRegistry.register("logistic")
class RegisteredLogisticKernel(LogisticKernel):
    """Registered logistic kernel function."""

    pass
