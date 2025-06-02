"""
Enhanced usage examples for the refactored PatternLocalExplainer.

This example demonstrates the key improvements including:
- Fluent interface
- Configuration management
- Registry system
- Enhanced error handling
- Backward compatibility
"""

import logging

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from patternlocal import PatternLocalExplainer, SimplificationRegistry, SolverRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)


def basic_usage_example():
    """Demonstrate basic usage with the enhanced architecture."""
    print("=== Basic Usage Example ===")

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with probability estimates enabled
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Verify model supports predict_proba
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba")

    def predict_fn(X):
        try:
            return model.predict_proba(X)
        except Exception as e:
            print(f"Error in predict_fn: {e}")
            raise

    # Create explainer with enhanced features (backward compatible)
    explainer = PatternLocalExplainer(
        simplification="lowrank",
        solver="local_covariance",
        simplification_params={"n_components": 10},
        solver_params={"k_ratio": 0.1},
        random_state=42,
    )

    # Fit and explain
    explainer.fit(X_train)

    # Test predict_fn before using it
    test_pred = predict_fn(X_test[0:1])
    print(f"Test prediction shape: {test_pred.shape}")

    explanation = explainer.explain_instance(X_test[0], predict_fn, X_train)

    print(f"Pattern weights shape: {explanation['pattern_weights'].shape}")
    print(f"LIME weights shape: {explanation['lime_weights'].shape}")
    print(f"Metadata: {explanation['metadata']}")


def fluent_interface_example():
    """Demonstrate the fluent interface."""
    print("\n=== Fluent Interface Example ===")

    # Create data
    X, y = make_classification(n_samples=500, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Use fluent interface to configure explainer
    explainer = (
        PatternLocalExplainer()
        .with_simplification("lowrank", n_components=8)
        .with_solver("local_covariance", k_ratio=0.15, shrinkage_intensity=0.1)
        .with_lime_params(num_samples=3000, feature_selection="auto")
        .fit(X_train)
    )

    print(f"Explainer info: {explainer.get_explainer_info()}")

    # Explain instance
    explainer.explain_instance(X_test[0], predict_fn, X_train)
    print(f"Explanation completed using {explainer.mode} mode")


def configuration_management_example():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Example ===")

    # Create configuration dict
    config = {
        "simplification": "lowrank",
        "solver": "local_covariance",
        "lime_params": {"num_samples": 2000, "feature_selection": "lasso_path"},
        "simplification_params": {"n_components": 0.95},
        "solver_params": {"k_ratio": 0.2},
        "random_state": 42,
    }

    # Load configuration and create explainer
    explainer = PatternLocalExplainer.from_config(config)
    print(f"Loaded explainer: {explainer}")


def registry_usage_example():
    """Demonstrate registry usage."""
    print("\n=== Registry Usage Example ===")

    # List available methods
    try:
        print(
            f"Available simplification methods: {
                SimplificationRegistry.list_available()}"
        )
    except BaseException:
        print("Registry not fully initialized (some methods not registered yet)")

    try:
        print(f"Available solver methods: {SolverRegistry.list_available()}")
    except BaseException:
        print("Registry not fully initialized (some methods not registered yet)")

    # Try to create methods using registry (with fallback to legacy)
    try:
        explainer = PatternLocalExplainer(
            simplification="lowrank", solver="local_covariance"
        )
        print(f"Created explainer: {explainer}")
    except Exception as e:
        print(f"Error creating explainer: {e}")


def batch_processing_example():
    """Demonstrate processing multiple instances."""
    print("\n=== Batch Processing Example ===")

    # Load real dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Create explainer
    explainer = PatternLocalExplainer(
        simplification="none",
        solver="local_covariance",
        solver_params={"k_ratio": 0.1},
        random_state=42,
    )

    explainer.fit(X_train)

    # Explain multiple instances sequentially
    batch_size = 5
    test_instances = X_test[:batch_size]

    print(f"Explaining {batch_size} instances...")
    explanations = []
    for i in range(batch_size):
        explanation = explainer.explain_instance(test_instances[i], predict_fn, X_train)
        explanations.append(explanation)

    print(f"Generated {len(explanations)} explanations")
    for i, explanation in enumerate(explanations):
        weights_range = (
            explanation["pattern_weights"].min(),
            explanation["pattern_weights"].max(),
        )
        print(f"Instance {i}: Pattern weights range {weights_range}")


def error_handling_example():
    """Demonstrate enhanced error handling."""
    print("\n=== Error Handling Example ===")

    try:
        # Try to create explainer with invalid configuration
        explainer = PatternLocalExplainer(
            simplification="invalid_method", solver="local_covariance"
        )
    except Exception as e:
        print(f"Caught configuration error: {type(e).__name__}: {e}")

    try:
        # Try to explain without fitting
        explainer = PatternLocalExplainer()
        X = np.random.rand(10, 5)
        explainer.explain_instance(X[0], lambda x: x[:, 0], X)
    except Exception as e:
        print(f"Caught validation error: {type(e).__name__}: {e}")


def backward_compatibility_example():
    """Demonstrate backward compatibility with original API."""
    print("\n=== Backward Compatibility Example ===")

    # Create data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Original API style still works
    explainer = PatternLocalExplainer(
        simplification="lowrank", solver="local_covariance"
    )

    explainer.fit(X_train)
    explanation = explainer.explain_instance(X_test[0], predict_fn, X_train)
    print("Backward compatible explanation completed")
    print(f"Original API still returns: {list(explanation.keys())}")


def main():
    """Run all examples."""
    print("Running PatternLocal Enhanced Usage Examples")
    print("=" * 50)

    basic_usage_example()
    fluent_interface_example()
    configuration_management_example()
    registry_usage_example()
    batch_processing_example()
    error_handling_example()
    backward_compatibility_example()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
