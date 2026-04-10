# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def clf_arrays():
    """Binary classification arrays matching data_processing output shape."""
    np.random.seed(42)
    n_train, n_test = 80, 20
    X_train = pd.DataFrame({"a": np.random.randn(n_train), "b": np.random.randn(n_train)})
    X_test  = pd.DataFrame({"a": np.random.randn(n_test),  "b": np.random.randn(n_test)})
    # y as DataFrame — same as data_processing produces (double-bracket)
    y_train = pd.DataFrame({"target": np.random.choice([0, 1], n_train)})
    y_test  = pd.DataFrame({"target": np.random.choice([0, 1], n_test)})
    return X_train, X_test, y_train, y_test


@pytest.fixture
def reg_arrays():
    """Regression arrays matching data_processing output shape."""
    np.random.seed(42)
    n_train, n_test = 80, 20
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + np.random.randn(100) * 0.05
    X_train = pd.DataFrame(X[:n_train], columns=["fa", "fb", "fc"])
    X_test  = pd.DataFrame(X[n_train:], columns=["fa", "fb", "fc"])
    y_train = pd.DataFrame(y[:n_train], columns=["price"])
    y_test  = pd.DataFrame(y[n_train:], columns=["price"])
    return X_train, X_test, y_train, y_test
