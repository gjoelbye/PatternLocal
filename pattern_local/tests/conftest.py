"""
Pytest configuration and shared fixtures for PatternLocal tests.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def random_state():
    """Fixed random state for reproducible tests."""
    return 42


@pytest.fixture
def small_tabular_data(random_state):
    """Small tabular dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )
    return X, y


@pytest.fixture
def small_image_data(random_state):
    """Small image dataset for testing."""
    image_shape = (8, 8)
    n_features = image_shape[0] * image_shape[1]
    X = np.random.RandomState(random_state).rand(50, n_features)
    y = np.random.RandomState(random_state).randint(0, 2, 50)
    return X, y, image_shape


@pytest.fixture
def trained_tabular_model(small_tabular_data, random_state):
    """Trained model on tabular data."""
    X, y = small_tabular_data
    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_image_model(small_image_data, random_state):
    """Trained model on image data."""
    X, y, _ = small_image_data
    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    model.fit(X, y)
    return model


@pytest.fixture
def predict_fn_tabular(trained_tabular_model):
    """Prediction function for tabular model."""

    def predict_fn(X):
        return trained_tabular_model.predict_proba(X)

    return predict_fn


@pytest.fixture
def predict_fn_image(trained_image_model):
    """Prediction function for image model."""

    def predict_fn(X):
        return trained_image_model.predict_proba(X)

    return predict_fn
