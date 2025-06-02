"""
Simplification methods for PatternLocal.

This module provides various data preprocessing/simplification methods
that can be applied before pattern computation.
"""

from .base import BaseSimplification
from .registry import SimplificationRegistry
from .no_simplification import NoSimplification
from .lowrank import LowRankSimplification
from .superpixel import SuperpixelSimplification

__all__ = [
    "BaseSimplification",
    "SimplificationRegistry",
    "NoSimplification", 
    "LowRankSimplification",
    "SuperpixelSimplification"
] 