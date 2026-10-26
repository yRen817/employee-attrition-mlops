
# Employee Attrition MLOps Pipeline

## Project Overview

This project builds an end-to-end MLOps pipeline to predict employee attrition using machine learning.

The goal is to demonstrate reproducible data processing, experiment tracking, automated testing, CI/CD, and data drift monitoring.

## Dataset

The project uses the IBM HR Analytics Employee Attrition & Performance dataset.

Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Prediction target: `Attrition`

Task: Binary classification to predict whether an employee leaves the company.

Missing values will be simulated in selected input features to demonstrate preprocessing and data quality practices.

## Technology Stack

- Python
- pandas
- scikit-learn
- Git and GitHub
- DVC
- MLflow
- pytest
- GitHub Actions
- Evidently

## Project Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Retrieve the dataset:

   `dvc pull`

## Training

Training instructions will be documented after the training pipeline is implemented.

## Testing

Run:

`pytest tests/ -v`

## Monitoring

The project will compare reference data against simulated production data using Evidently and generate a drift report.

## Reproducibility

The project will use configuration files, versioned data, recorded model parameters, and automated tests.

## Limitations

This is an educational project. Predictions are not intended for employment decisions or evaluating individual employees.
