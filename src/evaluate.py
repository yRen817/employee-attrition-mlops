
"""Evaluate the best MLflow model on the held-out test set."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import (
    MISSINGNESS_COLUMNS,
    load_data,
    simulate_missing_values,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking_uri = config["mlflow"]["tracking_uri"]
    if "://" not in tracking_uri:
        tracking_path = (PROJECT_ROOT / tracking_uri).resolve()
        tracking_uri = tracking_path.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )
    if experiment is None:
        raise RuntimeError("MLflow experiment was not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.validation_f1 DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError("No completed runs with validation metrics found.")

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]

    data_config = config["data"]
    data_path = Path(data_config["input_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    df = load_data(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column=data_config["target_column"],
        test_size=data_config["test_size"],
        validation_size=data_config["validation_size"],
        random_state=config["project"]["random_state"],
    )

    missing_rate = data_config.get("missing_rate", 0.05)
    random_state = config["project"]["random_state"]
    missing_columns = [
        column
        for column in MISSINGNESS_COLUMNS
        if column in X_test.columns
    ]

    X_test = simulate_missing_values(
        X_test,
        columns=missing_columns,
        missing_rate=missing_rate,
        random_state=random_state + 2,
    )

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, predictions, zero_division=0),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
    }

    print(f"Selected run: {run_id}")
    print(f"Validation F1: {best_run['metrics.validation_f1']:.4f}")
    print("\nFinal held-out test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
