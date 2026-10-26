
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    build_preprocessor,
    simulate_missing_values,
    split_data,
    validate_data,
)


@pytest.fixture
def sample_df():
    """Small synthetic employee dataset for unit tests."""
    n = 100

    return pd.DataFrame({
        "Age": np.tile([25, 30, 35, 40, 45], 20),
        "MonthlyIncome": np.arange(100) * 100 + 2000,
        "DistanceFromHome": np.arange(100) % 20 + 1,
        "Department": np.tile(
            ["Sales", "Research", "HR", "Sales", "Research"],
            20,
        ),
        "OverTime": np.tile(["Yes", "No"], 50),
        "Attrition": np.tile(["Yes", "No"], 50),
    })


def test_missing_value_simulation_adds_missing_values(sample_df):
    result = simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome"],
        missing_rate=0.20,
        random_state=42,
    )

    assert result["MonthlyIncome"].isna().sum() == 20


def test_missing_value_simulation_is_reproducible(sample_df):
    result1 = simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome"],
        missing_rate=0.20,
        random_state=42,
    )

    result2 = simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome"],
        missing_rate=0.20,
        random_state=42,
    )

    pd.testing.assert_frame_equal(result1, result2)


def test_missing_simulation_does_not_modify_original(sample_df):
    original = sample_df.copy(deep=True)

    simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome"],
        missing_rate=0.20,
        random_state=42,
    )

    pd.testing.assert_frame_equal(sample_df, original)


def test_missing_simulation_preserves_target(sample_df):
    result = simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome", "DistanceFromHome"],
        missing_rate=0.20,
        random_state=42,
    )

    assert result["Attrition"].notna().all()
    pd.testing.assert_series_equal(
        result["Attrition"],
        sample_df["Attrition"],
    )


def test_zero_missing_rate_returns_copy(sample_df):
    result = simulate_missing_values(
        sample_df,
        columns=["MonthlyIncome"],
        missing_rate=0,
    )

    pd.testing.assert_frame_equal(result, sample_df)
    assert result is not sample_df


def test_invalid_missing_rate_raises_error(sample_df):
    with pytest.raises(ValueError):
        simulate_missing_values(
            sample_df,
            missing_rate=1.0,
        )


def test_invalid_column_raises_error(sample_df):
    with pytest.raises(ValueError, match="Column not found"):
        simulate_missing_values(
            sample_df,
            columns=["UnknownFeature"],
        )


def test_invalid_target_raises_error(sample_df):
    bad_df = sample_df.copy()
    bad_df.loc[0, "Attrition"] = "Maybe"

    with pytest.raises(ValueError, match="Unexpected target"):
        validate_data(bad_df)


def test_missing_target_column_raises_error(sample_df):
    bad_df = sample_df.drop(columns=["Attrition"])

    with pytest.raises(ValueError, match="Missing target"):
        validate_data(bad_df)


def test_categorical_variables_are_encoded(sample_df):
    X = sample_df.drop(columns=["Attrition"])

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    # One-hot encoding should produce more columns than
    # the original categorical columns alone.
    assert transformed.shape[1] > X.shape[1]

    assert np.isfinite(transformed).all()


def test_unknown_category_does_not_crash(sample_df):
    X = sample_df.drop(columns=["Attrition"])

    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["Department"] = "NewDepartment"

    transformed = preprocessor.transform(new_row)

    assert transformed.shape[0] == 1


def test_split_data_has_correct_shapes(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(sample_df)
    )

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20

    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20


def test_split_data_encodes_target(sample_df):
    *_, y_train, y_val, y_test = split_data(sample_df)

    for labels in [y_train, y_val, y_test]:
        assert set(labels.unique()).issubset({0, 1})
