"""
Configuration classes for PatternLocal.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml

from ..exceptions import ConfigurationError
from ..simplification.base import BaseSimplification
from ..solvers.base import BaseSolver

LimeMode = Literal["tabular", "image"]


@dataclass
class LimeConfig:
    """Configuration for LIME parameters."""

    mode: LimeMode = "tabular"
    num_samples: int = 5000
    feature_selection: str = "auto"
    discretize_continuous: bool = True
    kernel_width: Optional[float] = None
    sample_around_instance: bool = True
    verbose: bool = False
    labels: Optional[list] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_samples <= 0:
            raise ConfigurationError("num_samples must be positive")

        if self.mode == "image" and self.labels is None:
            self.labels = [1]  # Default for binary classification

        # Set mode-specific defaults
        if self.mode == "image":
            if self.feature_selection == "auto":
                self.feature_selection = "none"
            if self.kernel_width is None:
                self.kernel_width = 0.25
        elif self.mode == "tabular":
            if self.kernel_width is None:
                pass  # LIME will auto-estimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LIME explainer."""
        config = {
            "feature_selection": self.feature_selection,
            "num_samples": self.num_samples,
            "verbose": self.verbose,
        }

        if self.mode == "tabular":
            config.update(
                {
                    "discretize_continuous": self.discretize_continuous,
                    "sample_around_instance": self.sample_around_instance,
                }
            )
            if self.kernel_width is not None:
                config["kernel_width"] = self.kernel_width
        else:  # image mode
            config.update({"kernel_width": self.kernel_width, "labels": self.labels})

        return config


@dataclass
class SimplificationConfig:
    """Configuration for simplification methods."""

    method: str = "none"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Import here to avoid circular imports
        from ..simplification.registry import SimplificationRegistry

        valid_methods = SimplificationRegistry.list_available()
        if self.method not in valid_methods:
            raise ConfigurationError(
                f"Invalid simplification method: {
                    self.method}. Valid: {valid_methods}"
            )


@dataclass
class SolverConfig:
    """Configuration for solver methods."""

    method: str = "local_covariance"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Import here to avoid circular imports
        from ..solvers.registry import SolverRegistry

        valid_methods = SolverRegistry.list_available()
        if self.method not in valid_methods:
            raise ConfigurationError(
                f"Invalid solver method: {self.method}. Valid: {valid_methods}"
            )


@dataclass
class ExplainerConfig:
    """Main configuration class for PatternLocalExplainer."""

    simplification: Union[str, BaseSimplification, SimplificationConfig] = "none"
    solver: Union[str, BaseSolver, SolverConfig] = "local_covariance"
    lime_params: Union[Dict[str, Any], LimeConfig] = field(default_factory=dict)
    simplification_params: Dict[str, Any] = field(default_factory=dict)
    solver_params: Dict[str, Any] = field(default_factory=dict)
    random_state: Optional[int] = None
    enable_caching: bool = True
    cache_size: int = 128
    enable_parallel: bool = False
    n_jobs: int = 1

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.validate()
        self._normalize_configs()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.cache_size <= 0:
            raise ConfigurationError("cache_size must be positive")

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ConfigurationError("n_jobs must be -1 or positive integer")

        if self.random_state is not None and self.random_state < 0:
            raise ConfigurationError("random_state must be non-negative")

    def _normalize_configs(self) -> None:
        """Normalize configuration objects."""
        # Normalize LIME config
        if isinstance(self.lime_params, dict):
            self.lime_params = LimeConfig(**self.lime_params)
        elif not isinstance(self.lime_params, LimeConfig):
            raise ConfigurationError("lime_params must be dict or LimeConfig")

        # Normalize simplification config
        if isinstance(self.simplification, str):
            self.simplification = SimplificationConfig(
                method=self.simplification, params=self.simplification_params
            )
        elif isinstance(self.simplification, dict):
            self.simplification = SimplificationConfig(**self.simplification)
        elif not isinstance(
            self.simplification, (BaseSimplification, SimplificationConfig)
        ):
            raise ConfigurationError("Invalid simplification configuration")

        # Normalize solver config
        if isinstance(self.solver, str):
            self.solver = SolverConfig(method=self.solver, params=self.solver_params)
        elif isinstance(self.solver, dict):
            self.solver = SolverConfig(**self.solver)
        elif not isinstance(self.solver, (BaseSolver, SolverConfig)):
            raise ConfigurationError("Invalid solver configuration")

        # Auto-detect LIME mode from simplification
        if hasattr(self.simplification, "method"):
            if (
                self.simplification.method == "superpixel"
                and self.lime_params.mode == "tabular"
            ):
                self.lime_params.mode = "image"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExplainerConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ExplainerConfig":
        """Create configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "ExplainerConfig":
        """Create configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "simplification": getattr(
                self.simplification, "method", self.simplification
            ),
            "solver": getattr(self.solver, "method", self.solver),
            "lime_params": (
                self.lime_params.to_dict()
                if hasattr(self.lime_params, "to_dict")
                else self.lime_params
            ),
            "simplification_params": getattr(
                self.simplification, "params", self.simplification_params
            ),
            "solver_params": getattr(self.solver, "params", self.solver_params),
            "random_state": self.random_state,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "enable_parallel": self.enable_parallel,
            "n_jobs": self.n_jobs,
        }

    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
