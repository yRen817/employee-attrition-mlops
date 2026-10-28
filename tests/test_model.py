
"""Model validation tests."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor


def make_model_data():
    """Create reproducible synthetic binary classification data."""
    rng = np.random.default_rng(42)
    n_samples = 500

    risk_score = rng.normal(size=n_samples)

    df = pd.DataFrame({
        "RiskScore": risk_score,
        "Department": np.where(
            risk_score > 0, "Sales", "Research"
        ),
    })

    target = (risk_score > 0).astype(int)

    return df, target


def train_test_model():
    """Build and train a model for validation."""
    X, y = make_model_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_prediction_type_and_shape():
    """Predictions must be a 1D array of binary labels."""
    model, X_test, _ = train_test_model()

    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.ndim == 1
    assert predictions.shape[0] == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_meets_minimum_f1():
    """The model should perform well on a known synthetic test set."""
    model, X_test, y_test = train_test_model()

    predictions = model.predict(X_test)

    score = f1_score(y_test, predictions)

    assert score >= 0.80, (
        f"Synthetic test F1 {score:.4f} "
        "is below the minimum of 0.80"
    )
