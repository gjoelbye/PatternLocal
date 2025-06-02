"""
Simplification methods for PatternLocal.

This module provides various data preprocessing/simplification methods
that can be applied before pattern computation.
"""

from .base import BaseSimplification
from .lowrank import LowRankSimplification
from .no_simplification import NoSimplification
from .registry import SimplificationRegistry
from .superpixel import SuperpixelSimplification

__all__ = [
    "BaseSimplification",
    "SimplificationRegistry",
    "NoSimplification",
    "LowRankSimplification",
    "SuperpixelSimplification",
]
