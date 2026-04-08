import sys
import logging
import argparse
import os  
from pathlib import Path

# Fix path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib # noqa: E402
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import requests # noqa: E402
from scipy.stats import chi2_contingency, ks_2samp # noqa: E402

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, model_dir: Path = None,
                 p_val_threshold: float = 0.05,
                 psi_threshold: float = 0.20):

        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        self.model_dir = model_dir

        self.exclude_cols = [
            'CustomerID',
            'Tenure',
            'Usage Frequency',
            'Subscription Type',
            'Contract Length',
            'Churn'
        ]

        self.model = self._load_model()

    def _load_model(self):
        """Load latest model for prediction drift"""
        if not self.model_dir:
            return None

        # List of potential paths to look for the model
        # 1. Root models folder (where you just pushed it)
        # 2. The symlink 'latest' folder
        # 3. Fallback via LATEST_VERSION.txt
        
        possible_paths = [
            self.model_dir / "model.pkl",  # Priority: The file you just pushed
            self.model_dir / "latest" / "model.pkl"
        ]

        # Try direct paths first
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading model from {path}")
                return joblib.load(path)

        # Fallback for Windows/Text-based versioning
        latest_txt = self.model_dir / "LATEST_VERSION.txt"
        if latest_txt.exists():
            try:
                with open(latest_txt, "r") as f:
                    version_path = f.read().strip()
                    # version_path might look like 'run_2026...'
                    actual_path = self.model_dir.parent / version_path / "model.pkl"
                    if actual_path.exists():
                        logger.info(f"Loading model via LATEST_VERSION.txt: {actual_path}")
                        return joblib.load(actual_path)
            except Exception as e:
                logger.warning(f"Failed to read LATEST_VERSION.txt: {e}")

        logger.warning("No model found! System will skip prediction drift check.")
        return None
    # -----------------------------
    # PSI Calculation
    # -----------------------------
    @staticmethod
    def calculate_psi(expected, actual, bins=10):
        try:
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())

            e_counts, _ = np.histogram(expected, bins=bins, range=(min_val, max_val))
            a_counts, _ = np.histogram(actual, bins=bins, range=(min_val, max_val))

            e_percents = (e_counts + 1e-6) / len(expected)
            a_percents = (a_counts + 1e-6) / len(actual)

            psi = np.sum((e_percents - a_percents) * np.log(e_percents / a_percents))
            return psi
        except Exception as e:
            logger.error(f"PSI error: {e}")
            return 0.0

    # -----------------------------
    # Feature Drift
    # -----------------------------
    def detect_numerical_drift(self, ref, curr, name):
        _, p_val = ks_2samp(ref.dropna(), curr.dropna())
        psi_val = self.calculate_psi(ref.dropna(), curr.dropna())

        is_drifted = (p_val < self.p_val_threshold and psi_val > 0.1) or (psi_val >= self.psi_threshold)

        status = "DRIFT" if is_drifted else "STABLE"
        logger.info(f"[{status}] {name:18} | KS p-val: {p_val:.4f} | PSI: {psi_val:.4f}")
        return is_drifted

    def detect_categorical_drift(self, ref, curr, name):
        try:
            all_cats = sorted(set(ref.astype(str)) | set(curr.astype(str)))

            ref_counts = ref.astype(str).value_counts().reindex(all_cats, fill_value=0)
            curr_counts = curr.astype(str).value_counts().reindex(all_cats, fill_value=0)

            contingency = np.array([ref_counts.values, curr_counts.values])

            _, p_val, _, _ = chi2_contingency(contingency + 1e-6)

            is_drifted = p_val < self.p_val_threshold
            status = "DRIFT" if is_drifted else "STABLE"

            logger.info(f"[{status}] {name:18} | Chi2 p-val: {p_val:.4f}")
            return is_drifted

        except Exception as e:
            logger.error(f"Categorical drift failed: {name} | {e}")
            return False

    # -----------------------------
    # Prediction Drift 
    # -----------------------------
    def detect_prediction_drift(self, df_ref, df_curr):
        if self.model is None:
            logger.warning("No model loaded → skip prediction drift")
            return False

        try:
            X_ref = df_ref.drop(columns=['Churn'], errors='ignore')
            X_curr = df_curr.drop(columns=['Churn'], errors='ignore')

            # Use probability (better than class)
            ref_pred = self.model.predict_proba(X_ref)[:, 1]
            curr_pred = self.model.predict_proba(X_curr)[:, 1]

            psi_val = self.calculate_psi(ref_pred, curr_pred)
            _, p_val = ks_2samp(ref_pred, curr_pred)

            is_drifted = (p_val < self.p_val_threshold and psi_val > 0.1) or (psi_val >= self.psi_threshold)

            status = "DRIFT" if is_drifted else "STABLE"
            logger.info(f"[{status}] PREDICTION DRIFT | KS p-val: {p_val:.4f} | PSI: {psi_val:.4f}")

            return is_drifted

        except Exception as e:
            logger.error(f"Prediction drift failed: {e}")
            return False

    # -----------------------------
    # Retrain Trigger
    # -----------------------------
    # Trong file src/monitoring/detect_drift.py

    def trigger_retraining(self, data_path):
        logger.warning(" Critical drift detected. Sending trigger to GitHub Actions...")
        
        repo = os.getenv("GITHUB_REPOSITORY", "chanhbui297/DSEB65A-Group04-customer-churn-predictor")
        token = os.getenv("TOKENFORMLOPS")

        if not token:
            logger.error("GITHUB_TOKEN not found. Cannot trigger retraining.")
            return

        url = f"https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/dispatches"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        response = requests.post(url, headers=headers, json={"ref": "main"})
        
        if response.status_code == 204:
            logger.info("Retraining trigger sent successfully!")
        else:
            logger.error(f"Failed to trigger: {response.status_code} - {response.text}")

    # -----------------------------
    # Main Run
    # -----------------------------
    def run(self, train_path: Path, test_path: Path):
        if not train_path.exists() or not test_path.exists():
            logger.critical("Missing data files")
            sys.exit(1)

        df_ref = pd.read_csv(train_path)
        df_curr = pd.read_csv(test_path)

        logger.info("=" * 80)
        logger.info(f"DRIFT CHECK: {train_path.name} vs {test_path.name}")
        logger.info("=" * 80)

        drifted_features = []

        for col in df_ref.columns:
            if col in self.exclude_cols or col not in df_curr.columns:
                continue

            if pd.api.types.is_numeric_dtype(df_ref[col]) and df_ref[col].nunique() > 10:
                if self.detect_numerical_drift(df_ref[col], df_curr[col], col):
                    drifted_features.append(col)
            else:
                if self.detect_categorical_drift(df_ref[col], df_curr[col], col):
                    drifted_features.append(col)

        # Prediction Drift
        pred_drift = self.detect_prediction_drift(df_ref, df_curr)

        return drifted_features, pred_drift


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift Monitoring Tool")

    parser.add_argument("--ref", help="Train dataset")
    parser.add_argument("--curr", help="Serving dataset")
    parser.add_argument("--model_dir", help="Model directory (for prediction drift)")

    BASE_DIR = Path(__file__).resolve().parents[2]
    DEFAULT_TRAIN = BASE_DIR / "data/raw/train.csv"
    DEFAULT_TEST = BASE_DIR / "data/raw/test.csv"
    DEFAULT_MODEL = BASE_DIR / "models"
    
    args = parser.parse_args()

    ref_path = Path(args.ref) if args.ref else DEFAULT_TRAIN
    curr_path = Path(args.curr) if args.curr else DEFAULT_TEST
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL

    detector = DriftDetector(model_dir=model_dir)

    drifted_features, pred_drift = detector.run(ref_path, curr_path)

    logger.info("-" * 80)

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if len(drifted_features) >= 3 or pred_drift:
        logger.error("CRITICAL DRIFT DETECTED")
        detector.trigger_retraining(curr_path)

    elif len(drifted_features) > 0:
        logger.warning(f"MINOR DRIFT: {drifted_features}")

    else:
        logger.info("SYSTEM STABLE")