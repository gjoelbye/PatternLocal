#!/usr/bin/env python3
"""
Example: PatternLocal Hyperparameter Optimization

This example demonstrates how to use the hyperparameter optimization
capabilities of PatternLocal to automatically find the best parameters
for your explainer setup.
"""

import logging

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from patternlocal import OptimizedPatternLocalExplainer
    from patternlocal.optimization.metrics import (
        create_combined_metric,
        create_fidelity_metric,
        create_stability_metric,
    )
    from patternlocal.optimization.search_spaces import (
        create_custom_search_space,
    )

    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Optimization module not available: {e}")
    OPTIMIZATION_AVAILABLE = False


def create_synthetic_dataset(n_samples=1000, n_features=20, n_informative=10):
    """Create a synthetic dataset for demonstration."""
    logger.info(
        f"Creating synthetic dataset with {n_samples} samples and {n_features} features"
    )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    return X, y


def create_ground_truth_masks(X, n_informative=10):
    """Create synthetic ground truth importance masks."""
    logger.info("Creating ground truth importance masks")

    n_samples, n_features = X.shape
    masks = np.zeros((n_samples, n_features), dtype=bool)

    # First n_informative features are important
    masks[:, :n_informative] = True

    # Add some noise - randomly make some features important/unimportant
    for i in range(n_samples):
        # Randomly flip some bits
        flip_indices = np.random.choice(n_features, size=2, replace=False)
        masks[i, flip_indices] = ~masks[i, flip_indices]

    return masks


def train_model(X_train, y_train):
    """Train a simple model for demonstration."""
    logger.info("Training model")

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    model.fit(X_train, y_train)
    return model


def basic_optimization_example():
    """Basic hyperparameter optimization example."""
    logger.info("=== Basic Optimization Example ===")

    # Create dataset
    X, y = create_synthetic_dataset(n_samples=500, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Split test set into validation and final test
    X_val, X_final_test, y_val, y_final_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    # Create ground truth masks for validation set
    masks_val = create_ground_truth_masks(X_val, n_informative=10)

    # Train model
    model = train_model(X_train, y_train)

    logger.info(
        f"Model accuracy: {accuracy_score(y_final_test, model.predict(X_final_test)):.3f}"
    )

    # Create optimized explainer
    explainer = OptimizedPatternLocalExplainer(
        cache_dir="optimization_cache", n_jobs=2, random_state=42  # Use 2 parallel jobs
    )

    # Run hyperparameter optimization
    logger.info("Starting hyperparameter optimization...")

    best_params = explainer.optimize_parameters(
        X_val=X_val,
        masks_val=masks_val,
        X_train=X_train,
        y_train=y_train,
        model=model,
        backend="grid",  # Use grid search for reproducibility
        max_evals=20,  # Limited for demo purposes
        simplification="lowrank",
        solver="local_covariance",
        use_cache=True,
        save_results=True,
    )

    logger.info(f"Best parameters found: {best_params}")

    # Set explainer to use best parameters
    explainer.set_best_params(best_params)
    explainer.fit(X_train)

    # Test the optimized explainer
    test_instance = X_final_test[0]
    explanation = explainer.explain_instance(
        test_instance, model.predict_proba, X_train
    )

    logger.info(f"Example explanation weights: {explanation['pattern_weights'][:5]}")

    # Get optimization summary
    summary = explainer.get_optimization_summary()
    if summary:
        logger.info(
            f"Optimization completed in \
                {summary['optimization_stats']['execution_time_seconds']:.2f}s"
        )
        logger.info(f"Best loss: {summary['best_loss']:.6f}")


def advanced_optimization_example():
    """Advanced optimization with custom metrics and search spaces."""
    logger.info("=== Advanced Optimization Example ===")

    # Create dataset
    X, y = create_synthetic_dataset(n_samples=300, n_features=15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_final_test, y_val, y_final_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    masks_val = create_ground_truth_masks(X_val, n_informative=8)

    # Train model
    model = train_model(X_train, y_train)

    # Create custom combined metric
    fidelity_metric = create_fidelity_metric(n_samples=500)
    stability_metric = create_stability_metric(n_runs=5)

    combined_metric = create_combined_metric(
        metrics=[fidelity_metric, stability_metric],
        weights=[0.7, 0.3],  # 70% fidelity, 30% stability
    )

    # Create custom search space that includes multiple methods
    search_space = create_custom_search_space(
        simplification_methods=["lowrank", "none"],
        solver_methods=["local_covariance", "global_covariance"],
        include_lime=True,
        backend="grid",
    )

    # Create explainer
    explainer = OptimizedPatternLocalExplainer(
        cache_dir="optimization_cache",
        n_jobs=1,  # Sequential for stability
        random_state=42,
    )

    # Run optimization with custom settings
    logger.info("Starting advanced optimization with custom metrics...")

    best_params = explainer.optimize_parameters(
        X_val=X_val,
        masks_val=masks_val,
        X_train=X_train,
        y_train=y_train,
        model=model,
        metric_fn=combined_metric,
        search_space=search_space,
        backend="grid",
        max_evals=16,  # Limited grid for demo
        use_cache=True,
        save_results=True,
    )

    logger.info(f"Best parameters: {best_params}")

    # Analyze optimization history
    history = explainer.get_optimization_history()
    if history:
        losses = [h["loss"] for h in history if h["status"] == "ok"]
        logger.info(f"Optimization tried {len(history)} parameter combinations")
        logger.info(f"Loss range: {min(losses):.6f} - {max(losses):.6f}")


def hyperopt_optimization_example():
    """Example using Hyperopt backend (if available)."""
    logger.info("=== Hyperopt Optimization Example ===")

    try:
        import hyperopt

        print(hyperopt.__version__)

        logger.info("Hyperopt is available, running advanced optimization")
    except ImportError:
        logger.warning("Hyperopt not available, skipping this example")
        return

    # Create smaller dataset for faster optimization
    X, y = create_synthetic_dataset(n_samples=200, n_features=12)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_final_test, y_val, y_final_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    masks_val = create_ground_truth_masks(X_val, n_informative=6)

    # Train model
    model = train_model(X_train, y_train)

    # Create explainer
    explainer = OptimizedPatternLocalExplainer(
        cache_dir="optimization_cache", n_jobs=2, random_state=42
    )

    # Use hyperopt with fidelity metric
    logger.info("Running Hyperopt optimization...")

    best_params = explainer.optimize_parameters(
        X_val=X_val,
        masks_val=masks_val,
        X_train=X_train,
        y_train=y_train,
        model=model,
        backend="hyperopt",
        max_evals=30,
        simplification="lowrank",
        solver="local_covariance",
        use_cache=True,
    )

    logger.info(f"Hyperopt found best parameters: {best_params}")

    # Plot optimization history if matplotlib is available
    explainer.plot_optimization_history(save_path="optimization_history.png")
    logger.info("Optimization history plot saved to optimization_history.png")


def main():
    """Run all optimization examples."""
    if not OPTIMIZATION_AVAILABLE:
        logger.error(
            "Optimization module not available. Please install required dependencies."
        )
        return

    logger.info("PatternLocal Hyperparameter Optimization Examples")
    logger.info("=" * 60)

    try:
        # Run examples
        basic_optimization_example()
        print("\n" + "=" * 60 + "\n")

        advanced_optimization_example()
        print("\n" + "=" * 60 + "\n")

        hyperopt_optimization_example()

        logger.info("All examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
