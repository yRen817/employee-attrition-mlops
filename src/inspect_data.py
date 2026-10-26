
from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Please upload the CSV to data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n--- Target Distribution ---")
    if "Attrition" in df.columns:
        print(df["Attrition"].value_counts(dropna=False))
    else:
        print("Attrition column not found.")

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())


if __name__ == "__main__":
    main()
