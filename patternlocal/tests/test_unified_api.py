"""
Test the unified API functionality for both tabular and image modes.
"""

import numpy as np
import pytest

from patternlocal import PatternLocalExplainer
from patternlocal.exceptions import ValidationError


class TestUnifiedAPI:
    """Test unified API for both tabular and image modes."""

    def test_auto_detection_tabular(
        self, small_tabular_data, predict_fn_tabular, random_state
    ):
        """Test auto-detection of tabular mode."""
        X, y = small_tabular_data

        explainer = PatternLocalExplainer(
            simplification="none", solver="local_covariance", random_state=random_state
        )

        explainer.fit(X)

        # Should auto-detect tabular mode
        assert explainer.mode == "tabular"
        assert explainer.simplification_method == "NoSimplification"

        # Should work without specifying mode
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn_tabular, X_train=X
        )

        assert "pattern_weights" in explanation
        assert "lime_weights" in explanation
        assert explanation["pattern_weights"].shape == (X.shape[1],)

    def test_auto_detection_image(
        self, small_image_data, predict_fn_image, random_state
    ):
        """Test auto-detection of image mode."""
        X, y, image_shape = small_image_data

        explainer = PatternLocalExplainer(
            simplification="superpixel",
            solver="local_covariance",
            simplification_params={
                "image_shape": image_shape,
                "method": "grid",
                "grid_rows": 2,
                "grid_cols": 2,
            },
            random_state=random_state,
        )

        explainer.fit(X, image_shape=image_shape)

        # Should auto-detect image mode
        assert explainer.mode == "image"
        assert explainer.simplification_method == "SuperpixelSimplification"

        # Should work without specifying mode
        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn_image, X_train=X, labels=[1]
        )

        assert "pattern_weights" in explanation
        assert "lime_weights" in explanation
        assert explanation["pattern_weights"].shape == (X.shape[1],)

    def test_explicit_mode_override(
        self, small_image_data, predict_fn_image, random_state
    ):
        """Test explicit mode specification overrides auto-detection."""
        X, y, image_shape = small_image_data

        explainer = PatternLocalExplainer(
            simplification="superpixel",
            solver="local_covariance",
            lime_params={"mode": "image"},  # Explicit override
            simplification_params={
                "image_shape": image_shape,
                "method": "grid",
                "grid_rows": 2,
                "grid_cols": 2,
            },
            random_state=random_state,
        )

        explainer.fit(X, image_shape=image_shape)

        assert explainer.mode == "image"

        explanation = explainer.explain_instance(
            instance=X[0], predict_fn=predict_fn_image, X_train=X, labels=[1]
        )

        assert explanation["pattern_weights"].shape == (X.shape[1],)

    @pytest.mark.parametrize("solver", ["none", "local_covariance", "lasso", "ridge"])
    def test_all_solvers_work_with_both_modes(
        self,
        small_tabular_data,
        small_image_data,
        predict_fn_tabular,
        predict_fn_image,
        solver,
        random_state,
    ):
        """Test that all solvers work with both tabular and image modes."""
        X_tab, y_tab = small_tabular_data
        X_img, y_img, image_shape = small_image_data

        solver_params = {"k_ratio": 0.2} if solver != "none" else {}

        # Test tabular mode
        explainer_tab = PatternLocalExplainer(
            simplification="none",
            solver=solver,
            solver_params=solver_params,
            random_state=random_state,
        )
        explainer_tab.fit(X_tab)

        explanation_tab = explainer_tab.explain_instance(
            X_tab[0], predict_fn_tabular, X_tab
        )

        assert explainer_tab.mode == "tabular"
        assert explanation_tab["pattern_weights"].shape == (X_tab.shape[1],)

        # Test image mode
        explainer_img = PatternLocalExplainer(
            simplification="superpixel",
            solver=solver,
            simplification_params={
                "image_shape": image_shape,
                "method": "grid",
                "grid_rows": 2,
                "grid_cols": 2,
            },
            solver_params=solver_params,
            random_state=random_state,
        )
        explainer_img.fit(X_img, image_shape=image_shape)

        explanation_img = explainer_img.explain_instance(
            X_img[0], predict_fn_image, X_img, labels=[1]
        )

        assert explainer_img.mode == "image"
        assert explanation_img["pattern_weights"].shape == (X_img.shape[1],)

    def test_consistent_api_across_modes(
        self,
        small_tabular_data,
        small_image_data,
        predict_fn_tabular,
        predict_fn_image,
        random_state,
    ):
        """Test that the API is consistent across modes."""
        X_tab, y_tab = small_tabular_data
        X_img, y_img, image_shape = small_image_data

        # Both explainers should have the same interface
        explainer_tab = PatternLocalExplainer(
            simplification="none", solver="local_covariance", random_state=random_state
        )

        explainer_img = PatternLocalExplainer(
            simplification="superpixel",
            solver="local_covariance",
            simplification_params={
                "image_shape": image_shape,
                "method": "grid",
                "grid_rows": 2,
                "grid_cols": 2,
            },
            random_state=random_state,
        )

        # Both should have the same methods and properties
        for attr in [
            "fit",
            "explain_instance",
            "mode",
            "simplification_method",
            "solver_method",
        ]:
            assert hasattr(explainer_tab, attr)
            assert hasattr(explainer_img, attr)

        # Fit both
        explainer_tab.fit(X_tab)
        explainer_img.fit(X_img, image_shape=image_shape)

        # Generate explanations
        explanation_tab = explainer_tab.explain_instance(
            X_tab[0], predict_fn_tabular, X_tab
        )
        explanation_img = explainer_img.explain_instance(
            X_img[0], predict_fn_image, X_img, labels=[1]
        )

        # Both should return the same structure
        expected_keys = {
            "pattern_weights",
            "lime_weights",
            "lime_intercept",
            "local_exp",
            "metadata",
        }
        assert set(explanation_tab.keys()) == expected_keys
        assert set(explanation_img.keys()) == expected_keys

    def test_error_handling(self, random_state):
        """Test proper error handling for invalid configurations."""
        # Should raise error for unfitted explainer
        explainer = PatternLocalExplainer(random_state=random_state)
        with pytest.raises(
            ValidationError, match="explain_instance requires fitted explainer"
        ):
            explainer.explain_instance(
                np.random.rand(5),
                lambda x: np.random.rand(len(x), 2),
                np.random.rand(10, 5),
            )
