#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge labeled data into training set")
    parser.add_argument("--train", default="data/raw/train.csv", help="Existing training CSV")
    parser.add_argument("--new", default="data/raw/new_dataset_labeled.csv", help="New labeled CSV")
    parser.add_argument("--output", default="data/raw/train.csv", help="Output merged CSV path")
    args = parser.parse_args()

    train_path = Path(args.train)
    new_path = Path(args.new)

    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}")
        return
    if not new_path.exists():
        logger.error(f"New labeled file not found: {new_path}")
        return

    df_train = pd.read_csv(train_path)
    df_new = pd.read_csv(new_path)

    if "Churn" not in df_new.columns:
        logger.error("New data missing 'Churn' column. Run simulate_labels.py first.")
        return

    common_cols = list(set(df_train.columns) & set(df_new.columns))
    logger.info(f" Merging on {len(common_cols)} common columns")

    if "CustomerID" in common_cols:
        dup_count = len(df_new[df_new["CustomerID"].isin(df_train["CustomerID"])])
        if dup_count > 0:
            logger.warning(f"Removing {dup_count} duplicate CustomerIDs from new data")
            df_new = df_new[~df_new["CustomerID"].isin(df_train["CustomerID"])]

    df_merged = pd.concat([df_train, df_new[common_cols]], ignore_index=True)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    logger.info(f"Merged: {len(df_train)} + {len(df_new)} = {len(df_merged)} rows")
    logger.info(f"Saved to {output_path}")
    logger.info(f"New Churn rate: {df_merged['Churn'].mean():.2%}")

if __name__ == "__main__":
    main()