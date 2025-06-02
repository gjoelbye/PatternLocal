"""
PatternLocal Demo: Unified Pattern-based Explanations

This script demonstrates how to use the new unified PatternLocal package
with different simplification methods and pattern solvers.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from patternlocal import PatternLocalExplainer


def create_synthetic_data():
    """Create synthetic classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def demo_basic_usage():
    """Demonstrate basic PatternLocal usage."""
    print("=== Basic PatternLocal Demo ===")

    # Create data and train model
    X_train, X_test, y_train, y_test = create_synthetic_data()
    model = train_model(X_train, y_train)

    # Create prediction function
    def predict_fn(X):
        return model.predict_proba(X)

    # Initialize PatternLocal explainer with default settings
    explainer = PatternLocalExplainer(
        simplification="none", solver="local_covariance", random_state=42
    )

    # Fit explainer
    explainer.fit(X_train)

    # Explain an instance
    instance = X_test[0]
    explanation = explainer.explain_instance(
        instance=instance, predict_fn=predict_fn, X_train=X_train
    )

    pred_result = predict_fn(instance.reshape(1, -1))
    print(f"Model prediction: {pred_result[0][1]:.3f}")  # Class 1 probability
    print(f"Pattern weights shape: {explanation['pattern_weights'].shape}")
    print(f"Top 5 pattern weights: {explanation['pattern_weights'][:5]}")
    print(f"Top 5 LIME weights: {explanation['lime_weights'][:5]}")
    print()


def demo_different_solvers():
    """Demonstrate different pattern solvers."""
    print("=== Different Solvers Demo ===")

    # Create data and train model
    X_train, X_test, y_train, y_test = create_synthetic_data()
    model = train_model(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Test different solvers
    solvers = ["none", "global_covariance", "local_covariance", "lasso", "ridge"]
    instance = X_test[0]

    results = {}

    for solver_name in solvers:
        print(f"Testing solver: {solver_name}")

        # Configure solver parameters
        solver_params = {}
        if solver_name in ["lasso", "ridge"]:
            solver_params["alpha"] = 0.1
        if solver_name == "local_covariance":
            solver_params["k_ratio"] = 0.1
            solver_params["shrinkage_intensity"] = 0.1

        explainer = PatternLocalExplainer(
            simplification="none",
            solver=solver_name,
            solver_params=solver_params,
            random_state=42,
        )

        explainer.fit(X_train)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train
        )

        results[solver_name] = explanation["pattern_weights"]
        print(
            f"  Weight magnitude: {
                np.linalg.norm(
                    explanation['pattern_weights']):.3f}"
        )

    print()
    return results


def demo_lowrank_simplification():
    """Demonstrate low-rank simplification with PCA."""
    print("=== Low-Rank Simplification Demo ===")

    # Create higher dimensional data
    X, y = make_classification(
        n_samples=500,
        n_features=50,  # Higher dimension
        n_informative=10,
        n_redundant=20,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = train_model(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Compare with and without low-rank simplification
    configs = [
        {"simplification": "none", "name": "No simplification"},
        {
            "simplification": "lowrank",
            "simplification_params": {"n_components": 10},
            "name": "Low-rank (10 components)",
        },
        {
            "simplification": "lowrank",
            "simplification_params": {"n_components": 0.95},
            "name": "Low-rank (95% variance)",
        },
    ]

    instance = X_test[0]

    for config in configs:
        print(f"Testing: {config['name']}")

        explainer = PatternLocalExplainer(
            simplification=config["simplification"],
            solver="local_covariance",
            simplification_params=config.get("simplification_params", {}),
            random_state=42,
        )

        explainer.fit(X_train)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train
        )

        print(
            f"  Pattern weights shape: {
                explanation['pattern_weights'].shape}"
        )
        print(
            f"  Weight magnitude: {
                np.linalg.norm(
                    explanation['pattern_weights']):.3f}"
        )

        if hasattr(explainer.simplification, "n_components_fitted"):
            print(
                f"  PCA components: {
                    explainer.simplification.n_components_fitted}"
            )

    print()


def demo_custom_parameters():
    """Demonstrate advanced parameter customization."""
    print("=== Custom Parameters Demo ===")

    X_train, X_test, y_train, y_test = create_synthetic_data()
    model = train_model(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Custom kernel function
    def custom_kernel(distances, bandwidth):
        """Custom exponential kernel."""
        return np.exp(-distances / bandwidth)

    # Advanced configuration
    lime_params = {
        "num_samples": 10000,  # More samples for better stability
        "feature_selection": "lasso_path",
        "discretize_continuous": False,
    }

    solver_params = {
        "k_ratio": 0.15,  # Use 15% of training data
        "kernel_function": custom_kernel,
        "shrinkage_intensity": 0.2,
        "distance_metric": "euclidean",
        "use_projection": True,
    }

    explainer = PatternLocalExplainer(
        simplification="none",
        solver="local_covariance",
        lime_params=lime_params,
        solver_params=solver_params,
        random_state=42,
    )

    explainer.fit(X_train)

    instance = X_test[0]
    explanation = explainer.explain_instance(
        instance=instance,
        predict_fn=predict_fn,
        X_train=X_train,
        num_samples=15000,  # Override default
    )

    print("Custom configuration results:")
    print(
        f"  Pattern weights magnitude: {
            np.linalg.norm(
                explanation['pattern_weights']):.3f}"
    )
    print(
        f"  LIME weights magnitude: {
            np.linalg.norm(
                explanation['lime_weights']):.3f}"
    )
    print(f"  Simplification method: {explainer.simplification_method}")
    print(f"  Solver method: {explainer.solver_method}")
    print()


def demo_real_dataset():
    """Demonstrate on a real dataset (Breast Cancer)."""
    print("=== Real Dataset Demo (Breast Cancer) ===")

    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = train_model(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Test with different configurations
    configs = [
        {"solver": "none", "name": "LIME only"},
        {"solver": "global_covariance", "name": "Global covariance"},
        {"solver": "local_covariance", "name": "Local covariance"},
        {"solver": "lasso", "solver_params": {"alpha": 0.01}, "name": "Lasso pattern"},
    ]

    instance = X_test[0]

    pred_result = predict_fn(instance.reshape(1, -1))
    print(f"Explaining instance with prediction: {pred_result[0][1]:.3f}")
    print(f"True label: {y_test[0]}")
    print()

    for config in configs:
        explainer = PatternLocalExplainer(
            simplification="none",
            solver=config["solver"],
            solver_params=config.get("solver_params", {}),
            random_state=42,
        )

        explainer.fit(X_train)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train
        )

        # Show top contributing features
        pattern_weights = explanation["pattern_weights"]
        top_indices = np.argsort(np.abs(pattern_weights))[-5:][::-1]

        print(f"{config['name']}:")
        for i, idx in enumerate(top_indices):
            feature_name = data.feature_names[idx]
            weight = pattern_weights[idx]
            print(f"  {i + 1}. {feature_name}: {weight:.3f}")
        print()


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_different_solvers()
    demo_lowrank_simplification()
    demo_custom_parameters()
    demo_real_dataset()

    print("=== Demo Complete ===")
    print("The PatternLocal package provides a unified interface for:")
    print("- Multiple simplification methods (None, LowRank, Superpixel)")
    print("- Multiple pattern solvers (None, Global/Local Covariance, Lasso, Ridge)")
    print("- Flexible parameter customization")
    print("- Easy integration with existing ML pipelines")
