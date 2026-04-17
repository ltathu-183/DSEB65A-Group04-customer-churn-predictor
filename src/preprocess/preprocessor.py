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

# -----------------------------
# Clean logical errors
# -----------------------------
def clean_logical_errors(df: pd.DataFrame) -> pd.DataFrame:
    initial_shape = df.shape[0]

    cond_age = (df['Age'] >= 18) & (df['Age'] <= 100)
    cond_tenure = df['Tenure'] >= 0
    cond_usage = df['Usage Frequency'] >= 0
    cond_calls = df['Support Calls'] >= 0
    cond_delay = df['Payment Delay'] >= 0
    cond_spend = df['Total Spend'] >= 0
    cond_interaction = (df['Last Interaction'] >= 0) & (df['Last Interaction'] <= 365)

    clean_df = df[
        cond_age &
        cond_tenure &
        cond_usage &
        cond_calls &
        cond_delay &
        cond_spend &
        cond_interaction
    ].copy()

    dropped_rows = initial_shape - clean_df.shape[0]
    logger.info(f"Removed {dropped_rows} illogical row(s)")

    return clean_df

# -----------------------------
# Standardize categorical values
# -----------------------------
def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Clean logical errors
    df = clean_logical_errors(df)

    # 2. Standardize categorical
    df = standardize_categories(df)

    # 3. Drop columns
    df = df.drop(
        ["CustomerID", "Tenure", "Usage Frequency", "Subscription Type", "Contract Length"],
        axis=1,
        errors="ignore"
    )

    # 4. Drop NA
    df = df.dropna()
    df = df.drop_duplicates()


    # 5. Type casting
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