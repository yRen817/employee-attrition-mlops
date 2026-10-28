
# Employee Attrition MLOps Pipeline

## Project Overview

This project demonstrates an end-to-end machine learning operations (MLOps) pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset.

The goal is to demonstrate reproducible data preprocessing, model training and evaluation, experiment tracking, automated testing, continuous integration, data versioning, and data drift monitoring.

This is an educational portfolio project, not a validated employee-retention decision system.

## Problem and Dataset

The project treats employee attrition prediction as a binary classification task. The target column is `Attrition`.

Dataset source: [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The preprocessing pipeline simulates missing values in selected features to demonstrate data-quality handling. The monitoring workflow also creates simulated production data to demonstrate how changes in feature distributions can be detected.

## Technology Stack

- Python and pandas
- scikit-learn
- DVC for dataset versioning
- MLflow for experiment tracking
- pytest for automated testing
- GitHub Actions for continuous integration
- Evidently for data drift monitoring
- YAML configuration for training settings

## Repository Structure

```text
.
├── .github/workflows/       # GitHub Actions pipeline
├── configs/train.yaml       # Training configuration
├── data/raw/                # DVC-tracked dataset pointer
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor_drift.py
├── tests/                   # Automated tests
├── reports/
│   └── drift_report.html    # Generated drift report
├── compare_experiments.py   # Experiment comparison
├── MONITORING.md            # Monitoring approach and results
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 for the intended project environment.

1. Clone the repository and enter its directory.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Retrieve the dataset using DVC:

   ```bash
   dvc pull
   ```

   The configured DVC remote must be accessible to your account. If access is not configured, `dvc pull` will fail until valid remote access is established.

## Run the Pipeline

Run the tests:

```bash
pytest tests/ -v
```

Run training:

```bash
python -m src.train
```

Run evaluation:

```bash
python -m src.evaluate
```

Training and evaluation require the project's dataset and dependencies to be available. See `configs/train.yaml` for configurable training settings.

## Experiment Tracking

The project integrates MLflow to record model-training experiments and evaluation information. Experiment records can be used to compare configurations and review model results.

Run `compare_experiments.py` to compare experiments when the required MLflow runs are available.

## Continuous Integration

The GitHub Actions workflow at `.github/workflows/mlops-pipeline.yml` is configured to run on pushes and pull requests targeting `main`. It installs dependencies, retrieves DVC data, runs automated tests, and executes model training and evaluation.

The workflow requires the DVC remote and any necessary credentials to be configured correctly in the GitHub repository settings.

## Data Drift Monitoring

The monitoring workflow compares reference data with simulated production data using Evidently and produces an HTML report at `reports/drift_report.html`.

The simulation deliberately changes selected feature distributions to demonstrate drift detection. The resulting drift report is therefore a demonstration of the monitoring process, not evidence of drift in a real production environment.

See `MONITORING.md` for the monitoring methodology, thresholds, results, and interpretation.

## Reproducibility and Limitations

The project uses a YAML configuration, DVC data versioning, automated tests, MLflow tracking, and GitHub Actions to support reproducibility.

Results may vary with dependency versions, dataset versions, and configuration changes. This is an educational project; its predictions should not be used to make employment decisions or evaluate individual employees. The model may reflect biases in historical data and has not been validated for real-world employment use.

## Data Drift Monitoring

This project uses Evidently to monitor changes in employee attrition model input features. Simulated production data is used to test drift detection, with a configurable threshold that triggers a warning when the drift share exceeds 30%.

The monitoring script generates an HTML report and identifies features that may have shifted.

**Documentation:** [Data Drift Monitoring Details](docs/data-drift-monitoring.md)

**Report:** [View the HTML Drift Report](reports/drift_report.html)

Run the monitor using `python -m src.monitor_drift`.
