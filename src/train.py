
"""Train and evaluate an employee attrition model with MLflow."""

import argparse
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    DEFAULT_DROP_COLUMNS,
    MISSINGNESS_COLUMNS,
    build_preprocessor,
    load_data,
    simulate_missing_values,
    split_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path):
    """Load training settings from YAML."""
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def get_data_version(data_path):
    """Return the DVC content hash when available."""
    data_path = Path(data_path)
    pointer_path = Path(str(data_path) + ".dvc")

    if pointer_path.exists():
        with pointer_path.open("r", encoding="utf-8") as file:
            pointer = yaml.safe_load(file)

        outputs = pointer.get("outs", []) if pointer else []

        if outputs and outputs[0].get("md5"):
            return f"dvc-md5:{outputs[0]['md5']}"

    # Fallback for local development before DVC tracking.
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def flatten_config(config, prefix=""):
    """Convert nested configuration values to MLflow parameters."""
    flattened = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_config(value, full_key))
        elif isinstance(value, list):
            flattened[full_key] = ",".join(map(str, value))
        elif value is None:
            flattened[full_key] = "None"
        else:
            flattened[full_key] = str(value)

    return flattened


def build_model(model_type, params, random_state):
    """Create the configured classification model."""
    model_type = model_type.lower()

    if model_type == "logistic_regression":
        allowed = {"C", "max_iter", "class_weight", "solver"}
        model_params = {
            key: value for key, value in params.items()
            if key in allowed
        }
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("class_weight", "balanced")
        model_params.setdefault("C", 1.0)

        return LogisticRegression(
            random_state=random_state,
            **model_params,
        )

    if model_type == "random_forest":
        allowed = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
        }
        model_params = {
            key: value for key, value in params.items()
            if key in allowed
        }
        model_params.setdefault("n_estimators", 100)
        model_params.setdefault("class_weight", "balanced")

        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **model_params,
        )

    raise ValueError(
        f"Unsupported model type: {model_type}. "
        "Choose logistic_regression or random_forest."
    )


def calculate_metrics(model, X, y):
    """Calculate binary classification metrics."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
    }


def configure_tracking_uri(uri):
    """Resolve a local MLflow tracking path."""
    if "://" in uri:
        return uri

    path = Path(uri)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def run_training(config_path, model_type=None, c=None,
                 n_estimators=None, run_name=None):
    """Load data, train, evaluate, and log an MLflow run."""
    config = load_config(config_path)

    project_cfg = config["project"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    evaluation_cfg = config["evaluation"]
    mlflow_cfg = config["mlflow"]
    preprocessing_cfg = config.get("preprocessing", {})

    random_state = int(project_cfg["random_state"])

    data_path = Path(data_cfg["input_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    df = load_data(data_path)
    data_version = get_data_version(data_path)

    # Split before fitting any preprocessing.
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(
            df,
            target_column=data_cfg["target_column"],
            test_size=float(data_cfg["test_size"]),
            validation_size=float(data_cfg["validation_size"]),
            random_state=random_state,
        )
    )

    # Simulate missingness reproducibly in each split.
    mask_columns = [
        col for col in MISSINGNESS_COLUMNS
        if col in X_train.columns
    ]
    missing_rate = float(data_cfg["missing_rate"])

    X_train = simulate_missing_values(
        X_train, mask_columns, missing_rate, random_state
    )
    X_val = simulate_missing_values(
        X_val, mask_columns, missing_rate, random_state + 1
    )
    X_test = simulate_missing_values(
        X_test, mask_columns, missing_rate, random_state + 2
    )

    drop_columns = config.get("features", {}).get(
        "drop_columns", DEFAULT_DROP_COLUMNS
    )

    model_params = dict(model_cfg.get("params", {}))
    selected_model_type = model_type or model_cfg["type"]

    # Optional CLI overrides make separate experiments easy.
    if c is not None:
        model_params["C"] = c
    if n_estimators is not None:
        model_params["n_estimators"] = n_estimators

    classifier = build_model(
        selected_model_type,
        model_params,
        random_state,
    )

    preprocessor = build_preprocessor(
        X_train,
        drop_columns=drop_columns,
    )

    # The complete pipeline accepts raw feature rows.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    experiment_name = mlflow_cfg["experiment_name"]
    tracking_uri = configure_tracking_uri(
        mlflow_cfg["tracking_uri"]
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log the configuration, including effective overrides.
        logged_config = dict(config)
        logged_config["model"] = {
            "type": selected_model_type,
            "params": model_params,
        }

        mlflow.log_params(flatten_config(logged_config))
        mlflow.log_param("data.version", data_version)
        mlflow.log_param("data.rows", len(df))
        mlflow.log_param("data.features_before_encoding", X_train.shape[1])
        mlflow.log_param("preprocessing.missing_rate", missing_rate)


        pipeline.fit(X_train, y_train)

        # Validation data determines whether the model is accepted.
        val_metrics = calculate_metrics(pipeline, X_val, y_val)

        mlflow.log_metrics({
            f"validation_{key}": value
            for key, value in val_metrics.items()
        })

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Data version: {data_version}")

        print("\nValidation metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")

        primary_metric = evaluation_cfg["primary_metric"]
        minimum_score = float(evaluation_cfg["minimum_f1"])
        achieved_score = val_metrics[primary_metric]

        print(
            f"\nAcceptance check: {primary_metric} = "
            f"{achieved_score:.4f}; minimum = {minimum_score:.4f}"
        )

        if achieved_score < minimum_score:
            raise RuntimeError(
                f"Model failed acceptance: {primary_metric} "
                f"{achieved_score:.4f} < {minimum_score:.4f}"
            )

        print("Model passed the validation acceptance check.")

        # Evaluate the test data for final reporting.
        # Do not use test metrics for model selection.
        test_metrics = calculate_metrics(pipeline, X_test, y_test)

        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        print("\nTest metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Log the complete fitted preprocessing + model pipeline.
        input_example = X_train.head(5)

        signature = infer_signature(
            input_example,
            pipeline.predict(input_example),
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

        return {
            "run_id": run.info.run_id,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
    )
    parser.add_argument("--c", type=float)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        model_type=args.model_type,
        c=args.c,
        n_estimators=args.n_estimators,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
