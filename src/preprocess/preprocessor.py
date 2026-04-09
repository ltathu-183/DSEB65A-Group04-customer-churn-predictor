import pandas as pd
import logging

import argparse
from pathlib import Path

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns
    df = df.drop(
        ["CustomerID", "Tenure", "Usage Frequency", "Subscription Type", "Contract Length"],
        axis=1,
        errors="ignore"
    )

    # Drop NA
    df = df.dropna()

    # Type casting
    df['Churn'] = df['Churn'].astype(int)
    df['Age'] = df['Age'].astype(int)
    df['Support Calls'] = df['Support Calls'].astype(int)
    df['Payment Delay'] = df['Payment Delay'].astype(int)
    df['Last Interaction'] = df['Last Interaction'].astype(int)
    df['Total Spend'] = df['Total Spend'].astype(float)

    # Category
    df['Gender'] = df['Gender'].astype('category')

    return df


def main(input_path, output_path):
    df = pd.read_csv(input_path)
    df_clean = preprocess(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.input, args.output)