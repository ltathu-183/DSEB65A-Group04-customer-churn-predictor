import sys
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime
import json
import logging.handlers  

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from scipy.stats import chi2_contingency, ks_2samp  # noqa: E402

# -----------------------------
# Logging Configuration
# -----------------------------
os.makedirs("logs", exist_ok=True)

# Setup root logger with RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.handlers.RotatingFileHandler(
        "logs/monitoring.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    
    # Format 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)

import yaml

def load_config(config_path: str = None) -> dict:
    """Load drift config with fallback to defaults"""
    DEFAULTS = {
        "drift": {
            "p_value_threshold": 0.05,
            "psi_threshold": 0.20,
            "psi_warning": 0.10, 
        },
        "paths": {
            "default_train": "data/raw/train.csv",
            "default_test": "data/raw/new_dataset.csv",
            "model_dir": "models/",
        },
        "alerting": {
            "critical_drift_count": 3,
            "enable_github_trigger": True,  
        }
    }
    
    if config_path is None:
        possible_paths = [
            Path(__file__).resolve().parent / "drift_config.yaml",  # monitoring/drift_config.yaml
            Path(__file__).resolve().parents[1] / "config" / "drift_config.yaml", # config/drift_config.yaml
            Path("config") / "drift_config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = str(p)
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            for section, values in DEFAULTS.items():
                if section not in config:
                    config[section] = values
                elif isinstance(values, dict):
                    for key, val in values.items():
                        config[section].setdefault(key, val)
            logger.info(f"Loaded config from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config: {e}")
            return DEFAULTS
        except Exception as e:
            logger.warning(f"Config load error: {e}, using defaults")
            return DEFAULTS
    
    logger.warning(f"Config not found, using defaults")
    return DEFAULTS
        
class DriftDetector:
    def __init__(self, model_dir: Path = None,
                 p_val_threshold: float = 0.05,
                 psi_threshold: float = 0.20,
                 config: dict = None): 

        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        self.model_dir = Path(model_dir) if model_dir else None
        self.config = config or {}

        self.exclude_cols = [
            'CustomerID', 'Tenure', 'Usage Frequency', 
            'Subscription Type', 'Contract Length', 'Churn'
        ]
        
        self.baseline_stats = None
        if self.model_dir:
            baseline_path = self.model_dir / "baseline_stats.json"
            if baseline_path.exists():
                try:
                    with open(baseline_path, "r") as f:
                        self.baseline_stats = json.load(f)
                    logger.info(f"Loaded baseline stats from {baseline_path}")
                except Exception as e:
                    logger.warning(f"Failed to load baseline stats: {e}")
      
        self.model = self._load_model()

    def _load_model(self):
        """Load latest model for prediction drift"""
        if not self.model_dir:
            return None

        possible_paths = [
            self.model_dir / "model.pkl",
            self.model_dir / "latest" / "model.pkl"
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading model from {path}")
                return joblib.load(path)

        # Fallback via LATEST_VERSION.txt
        latest_txt = self.model_dir / "LATEST_VERSION.txt"
        if latest_txt.exists():
            try:
                with open(latest_txt, "r") as f:
                    version_path = f.read().strip()
                    actual_path = self.model_dir.parent / version_path / "model.pkl"
                    if actual_path.exists():
                        logger.info(f"Loading model via LATEST_VERSION.txt: {actual_path}")
                        return joblib.load(actual_path)
            except Exception as e:
                logger.warning(f"Failed to read LATEST_VERSION.txt: {e}")

        logger.warning("No model found! System will skip prediction drift check.")
        return None
    
    def check_integrity(self, df: pd.DataFrame, name: str):
        issues = []
        
        # 1. Check null values
        null_counts = df.isnull().mean()
        high_nulls = null_counts[null_counts > 0.2]
        if not high_nulls.empty:
            issues.append(f"High missing values in: {high_nulls.to_dict()}")

        if self.baseline_stats:
            required_features = list(self.baseline_stats.keys())
        else:
            required_features = []
        missing = [c for c in required_features if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # 3. Check constant features
        for col in df.columns:
            if df[col].nunique() <= 1 and col not in self.exclude_cols:
                issues.append(f"Constant feature detected: {col}")

        if issues:
            for issue in issues:
                logger.error(f"[INTEGRITY FAIL] {name}: {issue}")
            return False
        
        logger.info(f"[INTEGRITY OK] {name} passed basic checks.")
        return True
    
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
        if self.baseline_stats and name in self.baseline_stats:
            bs = self.baseline_stats[name]
            if bs.get("type") == "numeric" and bs.get("std") and bs["std"] > 0:
                z = abs(curr.mean() - bs["mean"]) / (bs["std"] + 1e-8)
                if z > 5.0:
                    logger.warning(f" FAST-CHECK: {name} mean shifted {z:.2f}σ from baseline")

        ref_clean = ref.dropna()
        curr_clean = curr.dropna()
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return False
            
        _, p_val = ks_2samp(ref_clean, curr_clean)
        psi_val = self.calculate_psi(ref_clean, curr_clean)

        is_drifted = (p_val < self.p_val_threshold and psi_val > 0.1) or (psi_val >= self.psi_threshold)
        
        psi_warn = self.config.get("drift", {}).get("psi_warning", 0.10)
        if not is_drifted and psi_val >= psi_warn:
            logger.warning(f" {name:18} | PSI {psi_val:.4f} exceeds warning threshold ({psi_warn})")

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

            ref_pred = self.model.predict_proba(X_ref)[:, 1]
            curr_pred = self.model.predict_proba(X_curr)[:, 1]

            logger.info(f"Confidence mean: {curr_pred.mean():.4f} | std: {curr_pred.std():.4f}")

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
    def trigger_retraining(self, data_path):
        if not self.config.get("alerting", {}).get("enable_github_trigger", True):
            logger.info("GitHub trigger disabled in config, skipping auto-retrain")
            return

        logger.warning("Critical drift detected. Sending trigger to GitHub Actions...")
        
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

        try:
            response = requests.post(url, headers=headers, json={"ref": "main"}, timeout=30)
            if response.status_code == 204:
                logger.info("Retraining trigger sent successfully!")
            else:
                logger.error(f"Failed to trigger: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Trigger request failed: {e}")

    # -----------------------------
    # Main Run
    # -----------------------------
    def run(self, train_path: Path, test_path: Path):
        if not train_path.exists() or not test_path.exists():
            logger.critical("Missing data files")
            sys.exit(1)

        logger.info(f"Loading reference: {train_path} | Current: {test_path}")
        df_ref = pd.read_csv(train_path)
        df_curr = pd.read_csv(test_path)
        
        if not self.check_integrity(df_curr, "Serving Data"):
            return ["INTEGRITY_FAILURE"], False
        
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
            
           
        pred_drift = self.detect_prediction_drift(df_ref, df_curr)

        metrics_log = {
            "timestamp": datetime.now().isoformat(),
            "drifted_features": drifted_features,
            "prediction_drift": bool(pred_drift),
            "integrity_status": "PASS"
        }
        log_path = Path("logs/monitoring.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # load existing data safely
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except:
                data = []
        else:
            data = []

        # append new log
        data.append(metrics_log)

        # write back
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return drifted_features, pred_drift


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift Monitoring Tool")
    parser.add_argument("--ref", help="Train dataset")
    parser.add_argument("--curr", help="Serving dataset")
    parser.add_argument("--model_dir", help="Model directory")
    parser.add_argument("--config", help="Path to config file", default=None)  
    
    args = parser.parse_args()
    
    # Load config 
    config = load_config(args.config)
    
    BASE_DIR = Path(__file__).resolve().parents[1]
    ref_path = Path(args.ref) if args.ref else BASE_DIR / config["paths"]["default_train"]
    curr_path = Path(args.curr) if args.curr else BASE_DIR / config["paths"]["default_test"]
    model_dir = Path(args.model_dir) if args.model_dir else BASE_DIR / config["paths"]["model_dir"]
    
    detector = DriftDetector(
        model_dir=model_dir,
        p_val_threshold=config["drift"]["p_value_threshold"],
        psi_threshold=config["drift"]["psi_threshold"],
        config=config
    )
    
    drifted_features, pred_drift = detector.run(ref_path, curr_path)
    
    logger.info("-" * 80)

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if len(drifted_features) > 0:
        logger.warning(f"Drift detected in {len(drifted_features)} features: {drifted_features}")

    critical_count = config.get("alerting", {}).get("critical_drift_count", 3)
    if len(drifted_features) >= critical_count or pred_drift:
        logger.error("CRITICAL DRIFT DETECTED")
        detector.trigger_retraining(curr_path)
    else:
        logger.info("System stable. No retrain needed.")