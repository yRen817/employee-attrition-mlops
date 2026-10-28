# Data Drift Monitoring

## Purpose

This project uses Evidently to monitor changes in employee attrition model
input features by comparing a reference dataset with simulated production data.

## Method

- The reference dataset contains 70% of the shuffled feature rows.
- The remaining 30% is used to create the simulated production dataset.
- The target column (`Attrition`) and configured drop columns are excluded.
- Drift is evaluated using Evidently's `DataDriftPreset`.
- A feature is counted as drifted when its ValueDrift p-value is below 0.05.
- The drift share is the number of drifted features divided by the total
  number of monitored features.

## Simulated Drift

To demonstrate monitoring, the script increases `MonthlyIncome` by 25%
and increases `Age` by five years, capped at 60, when those columns exist.

## Threshold and Results

The configured drift-share threshold is 0.30 (30%).

In the recorded run, 13 of 30 features were flagged as drifted, giving a
drift share of 43.3%. This exceeded the configured threshold, so the
monitor printed a warning and exited with code 1.

These results come from simulated data and are not evidence of drift in
actual production data.

## Outputs

- Reference dataset: `data/processed/reference.csv`
- Simulated production dataset: `data/processed/production_simulated.csv`
- HTML report: `reports/drift_report.html`

## Running the Monitor

Activate the project's compatible Python environment and run:

```bash
python -m src.monitor_drift
1

```
