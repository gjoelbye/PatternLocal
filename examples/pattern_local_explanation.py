"""
PatternLocal Explanation Examples

This script demonstrates how to use PatternLocal for explaining model predictions,
with a focus on the PatternLocal methodology and terminology.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from patternlocal import PatternLocalExplainer
from patternlocal.utils.kernels import KernelRegistry


def create_tabular_data():
    """Create a synthetic tabular dataset for demonstration."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Convert to DataFrame for better interpretability
    X_df = pd.DataFrame(X, columns=feature_names)

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def demonstrate_pattern_local_explanation():
    """Demonstrate PatternLocal explanation on a tabular dataset."""
    print("=== PatternLocal Explanation Demo ===")

    # Create and prepare data
    X_train, X_test, y_train, y_test = create_tabular_data()
    model = train_model(X_train, y_train)

    # Convert DataFrames to numpy arrays for the explainer
    X_train_np = X_train.values

    def predict_fn(X):
        # Convert to DataFrame if needed for the model
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            X_df = pd.DataFrame(X, columns=X_train.columns)
            return model.predict_proba(X_df)
        return model.predict_proba(X)

    # Initialize PatternLocal explainer
    explainer = PatternLocalExplainer(
        simplification="none",  # No simplification for clear interpretation
        solver="local_covariance",  # Use local covariance for pattern discovery
        solver_params={
            "k_ratio": 0.1,  # Use 10% of training data for local patterns
            "kernel": "gaussian",  # Gaussian kernel for smooth weighting
            "shrinkage_intensity": 0.1,  # Moderate shrinkage for stability
            "use_projection": True,  # Enable projection for better pattern discovery
        },
        random_state=42,
    )

    # Fit the explainer to the training data
    explainer.fit(X_train_np)

    # Select an instance to explain
    instance_idx = 0
    instance = X_test.iloc[instance_idx]
    instance_np = instance.values

    # Get the model's prediction
    pred_proba = predict_fn(instance.values.reshape(1, -1))[0]
    true_label = y_test[instance_idx]

    print(f"\nExplaining instance {instance_idx}:")
    print(f"True label: {true_label}")
    print(f"Model prediction (class 1 probability): {pred_proba[1]:.3f}")

    # Generate PatternLocal explanation
    explanation = explainer.explain_instance(
        instance=instance_np, predict_fn=predict_fn, X_train=X_train_np
    )

    # Extract pattern weights and LIME weights
    pattern_weights = explanation["pattern_weights"]
    lime_weights = explanation["lime_weights"]

    # Create a DataFrame for better visualization of feature importance
    feature_importance = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "PatternLocal Weight": pattern_weights,
            "LIME Weight": lime_weights,
            "Absolute PatternLocal Weight": np.abs(pattern_weights),
        }
    )

    # Sort by absolute PatternLocal weight
    feature_importance = feature_importance.sort_values(
        "Absolute PatternLocal Weight", ascending=False
    )

    print("\nTop 5 most important features according to PatternLocal:")
    print(feature_importance.head().to_string(index=False))

    # Compare PatternLocal and LIME weights
    print("\nPatternLocal vs LIME comparison:")
    print(f"PatternLocal weights magnitude: {np.linalg.norm(pattern_weights):.3f}")
    print(f"LIME weights magnitude: {np.linalg.norm(lime_weights):.3f}")

    # Show explanation metadata
    print("\nExplanation metadata:")
    print(f"Simplification method: {explainer.simplification_method}")
    print(f"Solver method: {explainer.solver_method}")
    print(f"Available kernels: {KernelRegistry.list_available()}")


def demonstrate_pattern_local_with_simplification():
    """Demonstrate PatternLocal explanation with low-rank simplification."""
    print("\n=== PatternLocal with Low-Rank Simplification ===")

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
        {
            "name": "No simplification",
            "simplification": "none",
            "simplification_params": {},
        },
        {
            "name": "Low-rank (10 components)",
            "simplification": "lowrank",
            "simplification_params": {"n_components": 10},
        },
        {
            "name": "Low-rank (95% variance)",
            "simplification": "lowrank",
            "simplification_params": {"n_components": 0.95},
        },
    ]

    instance = X_test[0]

    for config in configs:
        print(f"\nTesting: {config['name']}")

        explainer = PatternLocalExplainer(
            simplification=config["simplification"],
            solver="local_covariance",
            simplification_params=config["simplification_params"],
            solver_params={"k_ratio": 0.1},
            random_state=42,
        )

        explainer.fit(X_train)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train
        )

        print(f"Pattern weights shape: {explanation['pattern_weights'].shape}")
        print(f"Weight magnitude: {np.linalg.norm(explanation['pattern_weights']):.3f}")

        if hasattr(explainer.simplification, "n_components_fitted"):
            print(f"PCA components: {explainer.simplification.n_components_fitted}")


def demonstrate_real_world_example():
    """Demonstrate PatternLocal on the Breast Cancer dataset."""
    print("\n=== PatternLocal on Breast Cancer Dataset ===")

    # Load breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize and convert to numpy arrays
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train)
    X_test_np = scaler.transform(X_test)
    feature_names = X_train.columns

    # Train model
    model = train_model(X_train_np, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Test different configurations
    configs = [
        {
            "name": "Local covariance",
            "solver": "local_covariance",
            "solver_params": {"k_ratio": 0.1},
        },
        {
            "name": "Global covariance",
            "solver": "global_covariance",
            "solver_params": {},
        },
        {"name": "Lasso pattern", "solver": "lasso", "solver_params": {"alpha": 0.01}},
    ]

    instance = X_test_np[0]

    pred_result = predict_fn(instance.reshape(1, -1))
    print(f"Explaining instance with prediction: {pred_result[0][1]:.3f}")
    print(f"True label: {y_test[0]}")

    for config in configs:
        print(f"\nTesting solver: {config['name']}")

        explainer = PatternLocalExplainer(
            simplification="none",
            solver=config["solver"],
            solver_params=config["solver_params"],
            random_state=42,
        )

        explainer.fit(X_train_np)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train_np
        )

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "PatternLocal Weight": explanation["pattern_weights"],
                "LIME Weight": explanation["lime_weights"],
                "Absolute PatternLocal Weight": np.abs(explanation["pattern_weights"]),
            }
        )

        # Show top features
        print("\nTop 3 most important features:")
        print(
            feature_importance.nlargest(3, "Absolute PatternLocal Weight").to_string(
                index=False
            )
        )


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_pattern_local_explanation()
    demonstrate_pattern_local_with_simplification()
    demonstrate_real_world_example()

    print("\n=== PatternLocal Explanation Demo Complete ===")
    print("Key aspects demonstrated:")
    print("- Basic PatternLocal explanation workflow")
    print("- PatternLocal with low-rank simplification")
    print("- Real-world application on medical data")
    print("- Comparison of different solvers and configurations")
    print("- Integration with scikit-learn pipelines")
