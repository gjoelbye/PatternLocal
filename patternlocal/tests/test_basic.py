"""
Basic tests for PatternLocal package.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from patternlocal import PatternLocalExplainer
from patternlocal.exceptions import ConfigurationError, ValidationError
from patternlocal.simplification import LowRankSimplification, NoSimplification
from patternlocal.solvers import LassoSolver, LocalCovarianceSolver, NoSolver


class TestPatternLocalExplainer:
    """Test the main PatternLocalExplainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_initialization(self):
        """Test basic initialization."""
        explainer = PatternLocalExplainer()
        assert explainer.simplification_method == "NoSimplification"
        assert explainer.solver_method == "LocalCovarianceSolver"

    def test_string_initialization(self):
        """Test initialization with string parameters."""
        explainer = PatternLocalExplainer(simplification="none", solver="lasso")
        assert explainer.simplification_method == "NoSimplification"
        assert explainer.solver_method == "LassoSolver"

    def test_object_initialization(self):
        """Test initialization with object instances."""
        simplification = NoSimplification()
        solver = LassoSolver()

        explainer = PatternLocalExplainer(simplification=simplification, solver=solver)
        assert explainer.simplification is simplification
        assert explainer.solver is solver

    def test_fit_and_explain(self, sample_data, trained_model):
        """Test basic fit and explain functionality."""
        X, y = sample_data
        model = trained_model

        def predict_fn(X):
            return model.predict_proba(X)  # Return full probability matrix

        explainer = PatternLocalExplainer(
            simplification="none", solver="local_covariance", random_state=42
        )

        # Fit explainer
        explainer.fit(X)
        assert explainer.is_fitted

        # Explain instance
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn, X_train=X
        )

        # Check explanation structure
        assert "pattern_weights" in explanation
        assert "lime_weights" in explanation
        assert "lime_intercept" in explanation
        assert "local_exp" in explanation

        # Check shapes
        assert explanation["pattern_weights"].shape == (X.shape[1],)
        assert explanation["lime_weights"].shape == (X.shape[1],)

    def test_different_solvers(self, sample_data, trained_model):
        """Test different solvers work."""
        X, y = sample_data
        model = trained_model

        def predict_fn(X):
            return model.predict_proba(X)  # Return full probability matrix

        solvers = ["none", "global_covariance", "local_covariance", "lasso", "ridge"]

        for solver_name in solvers:
            explainer = PatternLocalExplainer(
                simplification="none", solver=solver_name, random_state=42
            )

            explainer.fit(X)
            explanation = explainer.explain_instance(
                instance=X[0], predict_fn=predict_fn, X_train=X
            )

            assert explanation["pattern_weights"].shape == (X.shape[1],)

    def test_lowrank_simplification(self, sample_data, trained_model):
        """Test low-rank simplification."""
        X, y = sample_data
        model = trained_model

        def predict_fn(X):
            return model.predict_proba(X)  # Return full probability matrix

        explainer = PatternLocalExplainer(
            simplification="lowrank",
            solver="local_covariance",
            simplification_params={"n_components": 5},
            random_state=42,
        )

        explainer.fit(X)
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn, X_train=X
        )

        # Weights should be in original space
        assert explanation["pattern_weights"].shape == (X.shape[1],)
        assert explainer.simplification.n_components_ == 5

    def test_custom_parameters(self, sample_data, trained_model):
        """Test custom parameter passing."""
        X, y = sample_data
        model = trained_model

        def predict_fn(X):
            return model.predict_proba(X)  # Return full probability matrix

        lime_params = {"num_samples": 1000}
        solver_params = {"k_ratio": 0.2, "shrinkage_intensity": 0.1}

        explainer = PatternLocalExplainer(
            simplification="none",
            solver="local_covariance",
            lime_params=lime_params,
            solver_params=solver_params,
            random_state=42,
        )

        explainer.fit(X)
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn, X_train=X
        )

        assert explanation["pattern_weights"].shape == (X.shape[1],)

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ConfigurationError):
            PatternLocalExplainer(simplification="invalid_method")

        with pytest.raises(ConfigurationError):
            PatternLocalExplainer(solver="invalid_solver")

    def test_explain_before_fit(self, sample_data, trained_model):
        """Test error when explaining before fitting."""
        X, y = sample_data
        model = trained_model

        def predict_fn(X):
            return model.predict_proba(X)  # Return full probability matrix

        explainer = PatternLocalExplainer()

        with pytest.raises(
            ValidationError, match="explain_instance requires fitted explainer"
        ):
            explainer.explain_instance(instance=X[0], predict_fn=predict_fn, X_train=X)


class TestComponents:
    """Test individual components."""

    def test_no_simplification(self):
        """Test NoSimplification component."""
        simplification = NoSimplification()
        X = np.random.randn(10, 5)

        simplification.fit(X)
        assert simplification.is_fitted

        # Test transforms return unchanged data
        instance = X[0]
        transformed = simplification.transform_instance(instance)
        np.testing.assert_array_equal(transformed, instance)

        weights = np.random.randn(5)
        inverse_weights = simplification.inverse_transform_weights(weights)
        np.testing.assert_array_equal(inverse_weights, weights)

    def test_lowrank_simplification(self):
        """Test LowRankSimplification component."""
        simplification = LowRankSimplification({"n_components": 3})
        X = np.random.randn(50, 10)

        simplification.fit(X)
        assert simplification.is_fitted
        assert simplification.n_components_ == 3

        # Test transforms
        instance = X[0]
        transformed = simplification.transform_instance(instance)
        assert transformed.shape == (3,)

        # Test inverse transform
        weights_lr = np.random.randn(3)
        weights_original = simplification.inverse_transform_weights(weights_lr)
        assert weights_original.shape == (10,)

    def test_no_solver(self):
        """Test NoSolver component."""
        solver = NoSolver()

        lime_weights = np.random.randn(5)
        lime_intercept = 0.5
        instance = np.random.randn(5)
        X_train = np.random.randn(20, 5)

        pattern_weights = solver.solve(lime_weights, lime_intercept, instance, X_train)
        np.testing.assert_array_equal(pattern_weights, lime_weights)

    def test_local_covariance_solver(self):
        """Test LocalCovarianceSolver component."""
        solver = LocalCovarianceSolver({"k_ratio": 0.5})

        lime_weights = np.random.randn(5)
        lime_intercept = 0.5
        instance = np.random.randn(5)
        X_train = np.random.randn(20, 5)

        pattern_weights = solver.solve(lime_weights, lime_intercept, instance, X_train)
        assert pattern_weights.shape == lime_weights.shape
        assert not np.array_equal(pattern_weights, lime_weights)  # Should be different


if __name__ == "__main__":
    pytest.main([__file__])
