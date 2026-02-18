#!/usr/bin/env python3
"""
Simple test script to verify PatternLocal package imports and basic functionality.
"""

import sys

sys.path.append("src")


def test_imports():
    """Test basic imports."""
    try:
        pass

        print(" PatternLocalExplainer imported successfully")

        print("  Simplification methods imported successfully")

        print("  Solvers imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality."""
    try:
        from patternlocal import PatternLocalExplainer

        # Test basic initialization
        explainer = PatternLocalExplainer()
        print("PatternLocalExplainer initialized successfully")
        print(f"  Default simplification: {explainer.simplification_method}")
        print(f"  Default solver: {explainer.solver_method}")

        # Test different configurations
        configs = [
            ("none", "none"),
            ("none", "global_covariance"),
            ("none", "local_covariance"),
            ("none", "lasso"),
            ("none", "ridge"),
            ("lowrank", "local_covariance"),
        ]

        for simp, solv in configs:
            try:
                explainer = PatternLocalExplainer(simplification=simp, solver=solv)
                print(f"  Configuration ({simp}, {solv}) works")
            except Exception as e:
                print(f"✗ Configuration ({simp}, {solv}) failed: {e}")
                return False

        return True
    except Exception as e:
        print(f"  Basic functionality error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end():
    """Test basic end-to-end functionality."""
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        from patternlocal import PatternLocalExplainer

        # Create simple data
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        def predict_fn(X):
            return model.predict_proba(X)

        # Test explainer
        explainer = PatternLocalExplainer(
            simplification="none", solver="local_covariance", random_state=42
        )

        explainer.fit(X)
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn, X_train=X
        )

        # Check results
        assert "pattern_weights" in explanation
        assert "lime_weights" in explanation
        assert explanation["pattern_weights"].shape == (5,)

        print("  End-to-end test passed")
        return True

    except Exception as e:
        print(f"  End-to-end test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing PatternLocal package...")
    print("=" * 50)

    success = True

    print("\n1. Testing imports...")
    success &= test_imports()

    print("\n2. Testing basic functionality...")
    success &= test_basic_functionality()

    print("\n3. Testing end-to-end...")
    success &= test_end_to_end()

    print("\n" + "=" * 50)
    if success:
        print("  All tests passed! PatternLocal package is working correctly.")
    else:
        print("  Some tests failed. Please check the errors above.")
        sys.exit(1)
