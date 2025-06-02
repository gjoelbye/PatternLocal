"""
Example demonstrating PatternLocal explainer with unified API.

This example shows how PatternLocal automatically detects image mode
when using superpixel simplification, providing a unified experience
for both tabular and image data.
"""

import os

# Import PatternLocal
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from patternlocal import PatternLocalExplainer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def create_sample_image_data(
    n_samples=1000, image_shape=(28, 28), n_classes=2, random_state=42
):
    """Create sample 2D image data for demonstration."""
    np.random.seed(random_state)

    # Create flattened feature data
    n_features = image_shape[0] * image_shape[1]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(50, n_features // 10),
        n_redundant=min(10, n_features // 50),
        n_classes=n_classes,
        random_state=random_state,
    )

    # Normalize to image-like values [0, 1]
    X = (X - X.min(axis=1, keepdims=True)) / (
        X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8
    )

    return X, y


def train_image_model(X_train, y_train):
    """Train a simple model on image data."""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


def demo_unified_api():
    """Demonstrate PatternLocal's unified API for image data."""
    print("=== PatternLocal Unified API Demo ===")

    # Create sample image data
    image_shape = (28, 28)
    X, y = create_sample_image_data(n_samples=500, image_shape=image_shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Image shape: {image_shape}")

    # Train model
    model = train_image_model(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")

    # Create prediction function (LIME requires full probability matrix)
    def predict_fn(X):
        return model.predict_proba(X)

    # Initialize explainer - mode is auto-detected from simplification type!
    explainer = PatternLocalExplainer(
        simplification="superpixel",  # Auto-detects image mode
        solver="local_covariance",  # Pattern solver
        simplification_params={
            "image_shape": image_shape,  # Required for image processing
            "method": "slic",  # Use SLIC segmentation
            "n_segments": 50,  # Number of superpixels
            "compactness": 8,  # SLIC compactness parameter
            "sigma": 0,  # SLIC smoothing parameter
        },
        solver_params={
            "k_ratio": 0.1,  # Use 10% of training data for local estimation
            "shrinkage_intensity": 0.1,  # Regularization for covariance matrix
            "distance_metric": "euclidean",
        },
        random_state=42,
    )

    # Fit explainer
    print("\nFitting explainer...")
    explainer.fit(X_train, image_shape=image_shape)
    print(f"  Auto-detected mode: {explainer.mode}")
    print(f"  Simplification: {explainer.simplification_method}")
    print(f"  Solver: {explainer.solver_method}")
    print(f"  Number of superpixels: {explainer.simplification.n_superpixels}")

    # Select instance to explain
    instance = X_test[0]
    prediction = predict_fn(instance.reshape(1, -1))[0]
    true_label = y_test[0]

    print("\nExplaining instance:")
    print(f"True label: {true_label}")
    print(f"Predicted probabilities: {prediction}")

    # Generate explanation
    explanation = explainer.explain_instance(
        instance=instance,
        predict_fn=predict_fn,
        X_train=X_train,
        labels=[1],  # Explain class 1
        num_features=20,  # Show top 20 features
    )

    # Extract results
    pattern_weights = explanation["pattern_weights"]
    lime_weights = explanation["lime_weights"]
    lime_intercept = explanation["lime_intercept"]

    print("\nExplanation statistics:")
    print(
        f"Pattern weights range: [{
            pattern_weights.min():.3f}, {
            pattern_weights.max():.3f}]"
    )
    print(
        f"LIME weights range: [{
            lime_weights.min():.3f}, {
            lime_weights.max():.3f}]"
    )
    print(f"LIME intercept: {lime_intercept:.3f}")

    # Visualize results
    try:
        visualize_explanations(instance, pattern_weights, lime_weights, image_shape)
    except ImportError:
        print("Matplotlib not available for visualization")

    return explanation


def visualize_explanations(instance, pattern_weights, lime_weights, image_shape):
    """Visualize the original image and explanations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(instance.reshape(image_shape), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # LIME explanation
    lime_image = lime_weights.reshape(image_shape)
    im1 = axes[1].imshow(
        lime_image,
        cmap="RdBu",
        vmin=-np.abs(lime_image).max(),
        vmax=np.abs(lime_image).max(),
    )
    axes[1].set_title("LIME Explanation")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    # Pattern explanation
    pattern_image = pattern_weights.reshape(image_shape)
    im2 = axes[2].imshow(
        pattern_image,
        cmap="RdBu",
        vmin=-np.abs(pattern_image).max(),
        vmax=np.abs(pattern_image).max(),
    )
    axes[2].set_title("Pattern Explanation")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("unified_api_explanation.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Visualization saved as 'unified_api_explanation.png'")


def demo_mode_comparison():
    """Compare tabular and image modes side by side."""
    print("\n=== Mode Comparison Demo ===")

    # Create data for both modes
    X_tabular = np.random.rand(200, 20)
    y_tabular = np.random.randint(0, 2, 200)

    image_shape = (16, 16)
    X_image = np.random.rand(200, image_shape[0] * image_shape[1])
    y_image = np.random.randint(0, 2, 200)

    # Train models
    model_tab = RandomForestClassifier(n_estimators=20, random_state=42)
    model_tab.fit(X_tabular, y_tabular)

    model_img = RandomForestClassifier(n_estimators=20, random_state=42)
    model_img.fit(X_image, y_image)

    def predict_fn_tab(X):
        return model_tab.predict_proba(X)

    def predict_fn_img(X):
        return model_img.predict_proba(X)

    print("Tabular Mode (auto-detected):")
    # Tabular explainer
    explainer_tab = PatternLocalExplainer(
        simplification="lowrank",  # Auto-detects tabular mode
        solver="local_covariance",
        simplification_params={"n_components": 0.95},
        random_state=42,
    )
    explainer_tab.fit(X_tabular)
    print(f"    Mode: {explainer_tab.mode}")
    print(f"    Simplification: {explainer_tab.simplification_method}")

    explanation_tab = explainer_tab.explain_instance(
        X_tabular[0], predict_fn_tab, X_tabular
    )
    print(
        f"    Generated explanation with shape: {
            explanation_tab['pattern_weights'].shape}"
    )

    print("\nImage Mode (auto-detected):")
    # Image explainer
    explainer_img = PatternLocalExplainer(
        simplification="superpixel",  # Auto-detects image mode
        solver="local_covariance",
        simplification_params={
            "image_shape": image_shape,
            "method": "grid",
            "grid_rows": 4,
            "grid_cols": 4,
        },
        random_state=42,
    )
    explainer_img.fit(X_image, image_shape=image_shape)
    print(f"    Mode: {explainer_img.mode}")
    print(f"    Simplification: {explainer_img.simplification_method}")
    print(
        f"    Number of superpixels: {
            explainer_img.simplification.n_superpixels}"
    )

    explanation_img = explainer_img.explain_instance(
        X_image[0], predict_fn_img, X_image, labels=[1]
    )
    print(
        f"    Generated explanation with shape: {
            explanation_img['pattern_weights'].shape}"
    )


def demo_different_solvers():
    """Compare different solvers in image mode."""
    print("\n=== Solver Comparison Demo ===")

    # Create smaller dataset for faster comparison
    image_shape = (16, 16)
    X, y = create_sample_image_data(n_samples=200, image_shape=image_shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_image_model(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)

    # Test different solvers
    solvers = ["none", "local_covariance", "lasso", "ridge"]
    instance = X_test[0]

    for solver_name in solvers:
        print(f"\nTesting solver: {solver_name}")

        # Configure explainer (mode auto-detected from 'superpixel')
        explainer = PatternLocalExplainer(
            simplification="superpixel",
            solver=solver_name,
            simplification_params={
                "image_shape": image_shape,
                "method": "grid",  # Use grid for speed
                "grid_rows": 4,
                "grid_cols": 4,
            },
            solver_params={"k_ratio": 0.2} if solver_name != "none" else {},
            random_state=42,
        )

        # Fit and explain
        explainer.fit(X_train, image_shape=image_shape)
        explanation = explainer.explain_instance(
            instance=instance, predict_fn=predict_fn, X_train=X_train, labels=[1]
        )

        pattern_weights = explanation["pattern_weights"]
        print(
            f"    Mode: {
                explainer.mode} | Weights range: [{
                pattern_weights.min():.3f}, {
                pattern_weights.max():.3f}]"
        )


if __name__ == "__main__":
    # Run demos
    explanation = demo_unified_api()
    demo_mode_comparison()
    demo_different_solvers()

    print("\n=== Demo Complete ===")
    print("\n🎉 Key Features Demonstrated:")
    print("  Unified API works seamlessly for both tabular and image data")
    print("  Automatic mode detection based on simplification type")
    print("  No need to explicitly specify LIME mode in most cases")
    print("  All solvers work with both data types")
    print("  Consistent interface and parameters across modes")
    print("\nThe implementation successfully unifies image and tabular modes!")
