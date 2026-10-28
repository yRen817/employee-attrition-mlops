
"""Data validation tests for employee attrition."""

import pandas as pd
import pytest

from src.data_preprocessing import validate_data


@pytest.fixture
def valid_dataframe():
    return pd.DataFrame({
        "Age": [25, 35, 45, 50],
        "MonthlyIncome": [3000, 5000, 7000, 9000],
        "Department": [
            "Sales", "HR", "Research", "Sales"
        ],
        "Attrition": ["Yes", "No", "Yes", "No"],
    })


def test_valid_dataframe_passes_validation(valid_dataframe):
    """A valid dataset should pass validation."""
    validate_data(valid_dataframe)


def test_missing_target_column_is_rejected(valid_dataframe):
    """The dataset must contain its target column."""
    invalid_df = valid_dataframe.drop(columns=["Attrition"])

    with pytest.raises(ValueError):
        validate_data(invalid_df)


def test_invalid_target_value_is_rejected(valid_dataframe):
    """Only Yes and No are allowed for Attrition."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[0, "Attrition"] = "Maybe"

    with pytest.raises(ValueError):
        validate_data(invalid_df)


def test_missing_target_value_is_rejected(valid_dataframe):
    """The target must not contain missing values."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[0, "Attrition"] = None

    with pytest.raises(ValueError):
        validate_data(invalid_df)


def test_duplicate_columns_are_rejected(valid_dataframe):
    """Duplicate column names should not be accepted."""
    invalid_df = valid_dataframe.copy()
    invalid_df.columns = [
        "Age",
        "Age",
        "Department",
        "Attrition",
    ]

    with pytest.raises(ValueError):
        validate_data(invalid_df)


def test_empty_dataframe_is_rejected(valid_dataframe):
    """An empty dataset should fail validation."""
    invalid_df = valid_dataframe.iloc[0:0]

    with pytest.raises(ValueError):
        validate_data(invalid_df)
