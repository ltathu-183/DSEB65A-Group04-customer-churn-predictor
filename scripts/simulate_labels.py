#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def simulate_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn" in df.columns:
        logger.warning("Cột 'Churn' đã tồn tại. Bỏ qua giả lập.")
        return df

    mask_high_risk = pd.Series(False, index=df.index)
    
    if "Usage Frequency" in df.columns:
        mask_high_risk |= df["Usage Frequency"] < 5
    if "Tenure" in df.columns:
        mask_high_risk |= df["Tenure"] < 6
    if "Payment Delay" in df.columns:
        mask_high_risk |= df["Payment Delay"] > 7

    df["Churn"] = mask_high_risk.astype(int)

    noise_idx = np.random.choice(df.index, size=int(len(df) * 0.08), replace=False)
    df.loc[noise_idx, "Churn"] = 1 - df.loc[noise_idx, "Churn"]
    
    churn_rate = df["Churn"].mean()
    logger.info(f" Simulated labels added. Churn rate: {churn_rate:.2%}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Simulate Churn labels for new dataset")
    parser.add_argument("--input", default="data/raw/new_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/raw/new_dataset_labeled.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    df_labeled = simulate_labels(df)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    logger.info(f" Saved labeled data to {output_path}")

if __name__ == "__main__":
    main()