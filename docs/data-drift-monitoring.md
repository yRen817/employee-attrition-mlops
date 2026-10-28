# Data Drift Monitoring

## Purpose

This project uses Evidently to monitor changes in the input features of an employee attrition prediction model. It compares a reference dataset with simulated production data to demonstrate how feature drift can be detected.

## Method

- The shuffled dataset is divided into a reference dataset containing 70% of the feature rows and a simulated production dataset containing the remaining 30%.
- The target column (`Attrition`) and configured drop columns are excluded from drift analysis.
- Evidently's `DataDriftPreset` generates a data drift report.
- A feature is classified as drifted when its ValueDrift p-value is below 0.05, according to the configured monitoring rule.
- The drift share is calculated as the number of drifted features divided by the total number of monitored features.

## Simulated Drift

To test the monitoring pipeline, the script intentionally modifies two features in the simulated production dataset:

- **MonthlyIncome:** Increased by 25%.
- **Age:** Increased by five years, capped at 60.

These changes simulate shifts in input data that the monitoring system should identify.

## Drift Threshold and Results

The configured drift-share threshold is **0.30 (30%)**.

In the recorded run, 13 of the 30 monitored features were flagged as drifted, resulting in a drift share of **43.3%**. Because this exceeded the configured threshold, the monitor printed a warning and exited with status code `1`.

This demonstrates that the monitoring pipeline can detect substantial changes in simulated data and flag them when the configured threshold is exceeded.

**Important:** These results are based on simulated data. They do not demonstrate drift in actual production data or prove that model predictive performance has degraded.

## Output Files

- Reference dataset: `data/processed/reference.csv`
- Simulated production dataset: `data/processed/production_simulated.csv`
- HTML drift report: `reports/drift_report.html`

The HTML report provides a detailed view of the drift analysis.

## Running the Monitor

Activate the project's compatible Python environment and run:

```bash
python -m src.monitor_drift
```

An exit status of `1` is expected when the measured drift share exceeds the configured threshold, assuming the script is implemented to use that exit code. A status of `0` indicates successful completion without triggering the threshold.
