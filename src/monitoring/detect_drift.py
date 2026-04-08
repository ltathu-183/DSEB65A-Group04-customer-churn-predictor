import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, p_val_threshold: float = 0.05, psi_threshold: float = 0.20):
        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        
        # ALIGNMENT: Exclude columns dropped in preprocess.py or non-predictive
        self.exclude_cols = [
            'CustomerID', 
            'Tenure', 
            'Usage Frequency', 
            'Subscription Type', 
            'Contract Length',
            'Churn'
        ]

    @staticmethod
    def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI).
        Standard Industry Thresholds: <0.1: Stable | 0.1-0.25: Monitor | >0.25: Drift
        """
        try:
            # Define common range to align bins
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())

            e_counts, _ = np.histogram(expected, bins=bins, range=(min_val, max_val))
            a_counts, _ = np.histogram(actual, bins=bins, range=(min_val, max_val))

            # Laplace Smoothing (1e-6) to prevent division by zero or log(0)
            e_percents = (e_counts + 1e-6) / len(expected)
            a_percents = (a_counts + 1e-6) / len(actual)

            psi_values = (e_percents - a_percents) * np.log(e_percents / a_percents)
            return np.sum(psi_values)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

    def detect_numerical_drift(self, ref: pd.Series, curr: pd.Series, name: str) -> bool:
        """Hybrid check using KS-Test (Significance) and PSI (Magnitude)."""
        # Statistical test
        _, p_val = ks_2samp(ref.dropna(), curr.dropna())
        # Effect size test
        psi_val = self.calculate_psi(ref.dropna(), curr.dropna())
        
        # Flag drift if statistically significant AND practically relevant (PSI > 0.1)
        # OR if the shift is massive (PSI > threshold) regardless of p-value.
        is_drifted = (p_val < self.p_val_threshold and psi_val > 0.1) or (psi_val >= self.psi_threshold)
        
        status = "DRIFT" if is_drifted else "STABLE"
        logger.info(f"[{status}] {name:18} | KS p-val: {p_val:.4f} | PSI: {psi_val:.4f}")
        return is_drifted

    def detect_categorical_drift(self, ref: pd.Series, curr: pd.Series, name: str) -> bool:
        """Chi-Square contingency test with category alignment."""
        try:
            # Align unique categories from both sets (Handles new/missing labels)
            all_cats = sorted(set(ref.astype(str).unique()) | set(curr.astype(str).unique()))
            ref_counts = ref.astype(str).value_counts().reindex(all_cats, fill_value=0)
            curr_counts = curr.astype(str).value_counts().reindex(all_cats, fill_value=0)
            
            contingency_table = np.array([ref_counts.values, curr_counts.values])
            
            # Perform Chi2 with smoothing
            _, p_val, _, _ = chi2_contingency(contingency_table + 1e-6)
            is_drifted = p_val < self.p_val_threshold
            
            status = "DRIFT" if is_drifted else "STABLE"
            logger.info(f"[{status}] {name:18} | Chi2 p-val: {p_val:.4f} (Categorical)")
            return is_drifted
        except Exception as e:
            logger.error(f"Categorical drift detection failed for {name}: {e}")
            return False

    def run(self, train_path: Path, test_path: Path):
        """Main execution logic for drift detection."""
        if not train_path.exists() or not test_path.exists():
            logger.critical(f"Input files not found. Paths: {train_path}, {test_path}")
            sys.exit(1) # Critical failure for CI

        try:
            df_ref = pd.read_csv(train_path)
            df_curr = pd.read_csv(test_path)
            
            logger.info("-" * 85)
            logger.info(f"STARTING DRIFT ANALYSIS: {train_path.name} vs {test_path.name}")
            logger.info("-" * 85)

            drifted_features = []
            
            for col in df_ref.columns:
                # Skip target, IDs, and columns dropped during preprocessing
                if col in self.exclude_cols:
                    continue
                
                if col not in df_curr.columns:
                    logger.warning(f"Feature '{col}' missing in current data!")
                    continue
                
                # Direct to appropriate test based on data type
                if pd.api.types.is_numeric_dtype(df_ref[col]) and df_ref[col].nunique() > 10:
                    if self.detect_numerical_drift(df_ref[col], df_curr[col], col):
                        drifted_features.append(col)
                else:
                    if self.detect_categorical_drift(df_ref[col], df_curr[col], col):
                        drifted_features.append(col)

            return drifted_features

        except Exception as e:
            logger.critical(f"Unexpected error during drift analysis: {e}")
            sys.exit(1)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    # Path resolution relative to script location
    BASE_DIR = Path(__file__).resolve().parents[2]
    TRAIN_FILE = BASE_DIR / "data" / "raw" / "train.csv"
    TEST_FILE = BASE_DIR / "data" / "raw" / "new_dataset.csv"

    # Initialize detector with enterprise thresholds
    detector = DriftDetector(p_val_threshold=0.05, psi_threshold=0.20)
    drifted_list = detector.run(TRAIN_FILE, TEST_FILE)

    logger.info("-" * 85)
    if drifted_list:
        logger.warning(f"RESULT: Drift detected in {len(drifted_list)} features: {drifted_list}")
        # Note: In CI/CD, you might return sys.exit(0) but trigger an alert, 
        # or sys.exit(1) to hard-stop the pipeline.
        sys.exit(0) 
    else:
        logger.info("RESULT: No significant drift detected. Data is stable.")
        sys.exit(0)