"""Monitor feature drift between reference and simulated production data."""

from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "train.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_datasets(config):
    """Create reproducible reference and simulated production datasets."""
    data_path = ROOT / config["data"]["input_path"]
    target = config["data"].get("target_column", "Attrition")
    drop_columns = config.get("features", {}).get("drop_columns", [])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Check out your DVC data first."
        )

    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    # Monitor model input features only, never the target.
    features = df.drop(
        columns=[target] + [
            column for column in drop_columns if column in df.columns
        ],
        errors="ignore",
    )

    if features.empty:
        raise ValueError("No input features remain for drift monitoring.")

    # Make a reproducible split without overlapping rows.
    random_state = config["monitoring"].get("random_state", 42)
    shuffled = features.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    split_index = int(len(shuffled) * 0.70)
    reference = shuffled.iloc[:split_index].copy()
    production = shuffled.iloc[split_index:].copy()

    if reference.empty or production.empty:
        raise ValueError("Not enough rows to create both datasets.")

    # Simulate realistic changes in two numerical input features.
    if "MonthlyIncome" in production.columns:
        production["MonthlyIncome"] = (
            production["MonthlyIncome"] * 1.25
        )

    if "Age" in production.columns:
        production["Age"] = (production["Age"] + 5).clip(upper=60)

    monitoring = config["monitoring"]
    reference_path = ROOT / monitoring["reference_path"]
    production_path = ROOT / monitoring["production_path"]

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.parent.mkdir(parents=True, exist_ok=True)

    reference.to_csv(reference_path, index=False)
    production.to_csv(production_path, index=False)

    print(f"Reference dataset: {reference_path}")
    print(f"Production dataset: {production_path}")
    print(f"Features monitored: {len(features.columns)}")
    print(f"Reference rows: {len(reference)}")
    print(f"Production rows: {len(production)}")

    return reference, production


def find_drifted_columns(result):
    """Extract drifted features from Evidently 0.7.8 results."""
    drifted = set()

    def walk(item):
        if isinstance(item, dict):
            metric_id = item.get("metric_id", "")

            if metric_id.startswith("ValueDrift(column="):
                column = metric_id.removeprefix(
                    "ValueDrift(column="
                ).removesuffix(")")

                # Evidently's ValueDrift value is a p-value.
                # A p-value below 0.05 indicates detected drift.
                p_value = item.get("value")
                if (
                    isinstance(p_value, (int, float))
                    and p_value < 0.05
                ):
                    drifted.add(column)

            for value in item.values():
                walk(value)

        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(result)
    return sorted(drifted)


def main():
    config = load_config()
    monitoring = config["monitoring"]

    reference, production = prepare_datasets(config)

    # Generate Evidently's data drift report.
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=production,
    )

    report_path = ROOT / monitoring["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    result = snapshot.dict()
    drifted_columns = find_drifted_columns(result)

    total_features = len(reference.columns)
    drift_share = len(drifted_columns) / total_features
    threshold = float(monitoring["drift_share_threshold"])

    print("\n--- Drift Monitoring Summary ---")
    print(f"Total input features: {total_features}")
    print(f"Drifted features: {len(drifted_columns)}")
    print(f"Drift share: {drift_share:.1%}")
    print(f"Configured threshold: {threshold:.1%}")
    print(
        "Drifted feature names: "
        + (", ".join(drifted_columns) if drifted_columns else "None detected")
    )
    print(f"HTML report: {report_path}")

    if drift_share > threshold:
        print("WARNING: Feature drift exceeds the configured threshold.")
        raise SystemExit(1)

    print("Drift share is within the configured threshold.")


if __name__ == "__main__":
    main()
