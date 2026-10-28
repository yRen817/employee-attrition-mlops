
"""Compare MLflow experiments using validation F1."""

from pathlib import Path

import mlflow
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking_uri = config["mlflow"]["tracking_uri"]
    if "://" not in tracking_uri:
        tracking_path = (PROJECT_ROOT / tracking_uri).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = tracking_path.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = config["mlflow"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' was not found. "
            "Run at least one training experiment first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.validation_f1 DESC"],
    )

    if runs.empty:
        print("No completed training runs were found.")
        return

    metric_columns = [
        "metrics.validation_f1",
        "metrics.validation_precision",
        "metrics.validation_recall",
        "metrics.validation_roc_auc",
        "metrics.validation_accuracy",
    ]

    available_columns = [
        column for column in metric_columns if column in runs.columns
    ]

    display_columns = [
        column
        for column in [
            "tags.mlflow.runName",
            "run_id",
            "params.model.type",
            "params.model.params.C",
            "params.model.params.n_estimators",
        ]
        if column in runs.columns
    ] + available_columns

    results = runs[display_columns].copy()
    results = results.rename(
        columns={
            "tags.mlflow.runName": "experiment",
            "run_id": "run_id",
            "params.model.type": "model_type",
            "params.model.params.C": "C",
            "params.model.params.n_estimators": "n_estimators",
            "metrics.validation_f1": "validation_f1",
            "metrics.validation_precision": "validation_precision",
            "metrics.validation_recall": "validation_recall",
            "metrics.validation_roc_auc": "validation_roc_auc",
            "metrics.validation_accuracy": "validation_accuracy",
        }
    )

    print("\nExperiments ranked by validation F1:\n")
    print(results.to_string(index=False))

    best = runs.iloc[0]
    print("\nBest run by validation F1:")
    print(f"Run name: {best.get('tags.mlflow.runName', 'unnamed')}")
    print(f"Run ID: {best['run_id']}")
    print(f"Validation F1: {best['metrics.validation_f1']:.4f}")


if __name__ == "__main__":
    main()
