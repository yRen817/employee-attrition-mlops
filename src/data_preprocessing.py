
"""Data loading, validation, splitting, and preprocessing."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Attrition"

DEFAULT_DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

MISSINGNESS_COLUMNS = [
    "MonthlyIncome",
    "DistanceFromHome",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the employee attrition CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate the basic structure and target values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    valid_targets = {"Yes", "No"}
    actual_targets = set(df[TARGET_COLUMN].unique())

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Unexpected target values: {actual_targets}"
        )

    if len(actual_targets) < 2:
        raise ValueError(
            "Target must contain both classes for stratified splitting."
        )


def simulate_missing_values(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reproducibly simulate missing values in selected input features.

    The original DataFrame is never modified.
    Existing missing values are preserved.
    The target column cannot be masked.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be >= 0 and < 1.")

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if columns is None:
        columns = MISSINGNESS_COLUMNS

    if not columns:
        return df.copy(deep=True)

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns in missingness list.")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        if col == TARGET_COLUMN:
            raise ValueError("Cannot simulate missing target values.")

    result = df.copy(deep=True)

    if missing_rate == 0:
        return result

    rng = np.random.default_rng(random_state)

    for col in columns:
        # Mask only currently observed values.
        eligible = np.flatnonzero(
            result[col].notna().to_numpy()
        )

        n_missing = int(round(len(eligible) * missing_rate))

        if n_missing == 0:
            continue

        selected = rng.choice(
            eligible,
            size=n_missing,
            replace=False,
        )

        result.iloc[
            selected, result.columns.get_loc(col)
        ] = np.nan

    return result


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split into train, validation, and test sets.

    test_size and validation_size are proportions of the
    original dataset.
    """
    validate_data(df)

    if target_column != TARGET_COLUMN:
        raise ValueError(
            f"This project expects target '{TARGET_COLUMN}'."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target encoding failed.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_relative_size = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(
    X_train: pd.DataFrame,
    drop_columns: Optional[list[str]] = None,
) -> ColumnTransformer:
    """Create preprocessing using training feature types."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a DataFrame.")

    if X_train.empty:
        raise ValueError("X_train cannot be empty.")

    drop_columns = (
        DEFAULT_DROP_COLUMNS
        if drop_columns is None
        else drop_columns
    )

    X_train = X_train.drop(
        columns=drop_columns,
        errors="ignore",
    )

    if X_train.shape[1] == 0:
        raise ValueError("No usable features remain.")

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    unsupported = set(X_train.columns) - set(
        numeric_features + categorical_features
    )

    if unsupported:
        raise ValueError(
            f"Unsupported feature types: {unsupported}"
        )

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    missing_rate: float = 0.05,
    random_state: int = 42,
    drop_columns: Optional[list[str]] = None,
):
    """
    Prepare data and fit preprocessing on training data only.

    Returns transformed splits, labels, and the fitted
    preprocessor.
    """
    validate_data(df)

    # Split first so we can simulate missingness separately
    # inside each split.
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(
            df,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
        )
    )

    # Simulate missingness only in eligible columns.
    mask_columns = [
        c for c in MISSINGNESS_COLUMNS
        if c in X_train.columns
    ]

    X_train = simulate_missing_values(
        X_train,
        columns=mask_columns,
        missing_rate=missing_rate,
        random_state=random_state,
    )

    X_val = simulate_missing_values(
        X_val,
        columns=mask_columns,
        missing_rate=missing_rate,
        random_state=random_state + 1,
    )

    X_test = simulate_missing_values(
        X_test,
        columns=mask_columns,
        missing_rate=missing_rate,
        random_state=random_state + 2,
    )

    if drop_columns is None:
        drop_columns = DEFAULT_DROP_COLUMNS

    X_train = X_train.drop(
        columns=drop_columns, errors="ignore"
    )
    X_val = X_val.drop(
        columns=drop_columns, errors="ignore"
    )
    X_test = X_test.drop(
        columns=drop_columns, errors="ignore"
    )

    preprocessor = build_preprocessor(
        X_train,
        drop_columns=[],
    )

    # Fit only on training features.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform validation/test without refitting.
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
    }
