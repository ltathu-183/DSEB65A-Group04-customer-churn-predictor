import sys
import os
import argparse
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, roc_auc_score)  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from src.features.engineer import FeatureEngineer, create_preprocessor  # noqa: E402
from src.preprocess.preprocessor import preprocess  # noqa: E402

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# Config Loading (Lean version)
# -----------------------------
def load_config(config_path: str = None) -> dict:
    """Load config với fallback defaults - chỉ lấy fields thực sự dùng"""
    defaults = {
        "model_artifacts": {
            "output_dir": "models/",
            "version_format": "run_{timestamp}",
            "keep_versions": 5,
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,
            "scoring": "f1",
        }
    }
    
    if config_path is None:
        possible_paths = [
            Path(__file__).resolve().parents[2] / "config" / "drift_config.yaml",
            Path("config") / "drift_config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = str(p)
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            for section in defaults:
                if section not in config:
                    config[section] = defaults[section]
                elif isinstance(defaults[section], dict):
                    for k, v in defaults[section].items():
                        config[section].setdefault(k, v)
            return config
        except Exception as e:
            logger.warning(f"Config load error: {e}, using defaults")
    
    return defaults


# -----------------------------
# Helper Functions (Lean)
# -----------------------------
def save_baseline_stats(df: pd.DataFrame, save_dir: Path, exclude_cols: list = None) -> Path:
    """
    Lưu baseline statistics của training data để hỗ trợ detect_drift.py:
    - Fast sanity check (z-score mean shift)
    - Audit/debug nhanh không cần load CSV lớn
    - Documentation cho non-technical stakeholder
    """
    if exclude_cols is None:
        exclude_cols = ['CustomerID', 'Tenure', 'Usage Frequency', 'Subscription Type', 'Contract Length', 'Churn']
        
    stats = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "null_ratio": float(df[col].isna().mean()),
                "type": "numeric"
            }
        else:
            stats[col] = {
                "categories": df[col].astype(str).dropna().unique().tolist()[:50],
                "n_unique": int(df[col].nunique()),
                "null_ratio": float(df[col].isna().mean()),
                "type": "categorical"
            }

    baseline_path = Path(save_dir) / "baseline_stats.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"Saved baseline stats: {baseline_path} ({len(stats)} features)")
    return baseline_path


def create_version_dir(base_dir: Path, format_str: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = format_str.format(timestamp=timestamp, run_id=os.getenv("GITHUB_RUN_ID", "local"))
    version_dir = Path(base_dir) / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def update_latest_pointer(base_dir: Path, version_dir: Path):
    base_dir = Path(base_dir)
    version_dir = Path(version_dir)

    critical_files = ["model.pkl", "metadata.json", "baseline_stats.json"]

    # Copy lightweight artifacts for quick access
    for fname in critical_files:
        src = version_dir / fname
        dst = base_dir / fname
        if src.exists():
            try:
                shutil.copy2(src, dst)
                logger.debug(f"Copied {fname} to {dst}")
            except Exception as e:
                logger.warning(f"Failed copying {fname} to {dst}: {e}")

    # Write LATEST_VERSION.txt using the version directory name (best-effort)
    try:
        (base_dir / "LATEST_VERSION.txt").write_text(str(version_dir.name))
    except Exception as e:
        logger.warning(f"Could not update LATEST_VERSION.txt: {e}")

    latest_path = base_dir / "latest"
    # Remove any existing target safely
    try:
        if latest_path.is_symlink():
            latest_path.unlink()
        elif latest_path.exists():
            if latest_path.is_dir():
                shutil.rmtree(latest_path)
            else:
                latest_path.unlink()
    except Exception as e:
        logger.warning(f"Could not remove existing 'latest': {e}")

    # Create a relative symlink pointing to the version directory name
    try:
        # Use just the folder name so symlink inside `models/` resolves to `models/run_xxx`
        target = version_dir.name
        latest_path.symlink_to(target, target_is_directory=True)
        logger.info(f"Created relative 'latest' symlink -> {target}")
    except Exception as e:
        logger.warning(f"Could not create relative symlink for 'latest': {e}. Falling back to folder copy.")
        try:
            latest_path.mkdir(parents=True, exist_ok=True)
            for fname in critical_files:
                src = version_dir / fname
                if src.exists():
                    try:
                        shutil.copy2(src, latest_path / fname)
                    except Exception as e2:
                        logger.warning(f"Failed copying {fname} into fallback latest: {e2}")
            logger.info("Used folder fallback for 'latest' (symlink not supported)")
        except Exception as e3:
            logger.error(f"Failed to set up 'latest' pointer: {e3}")


def cleanup_old_versions(base_dir: Path, keep_n: int):
    """Giữ lại N version mới nhất, xóa các version cũ"""
    versions = sorted(
        [d for d in Path(base_dir).iterdir() 
        if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.name, reverse=True
    )
    for old in versions[keep_n:]:
        try:
            shutil.rmtree(old)
            logger.info(f"🗑️ Cleaned: {old.name}")
        except Exception as e:
            logger.error(f"Failed to delete {old.name}: {e}")


# -----------------------------
# Main Training Logic
# -----------------------------
def main(args):
    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)
    
    # 1. Load config
    config = load_config(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(config["model_artifacts"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load data
    logger.info(f"Loading data from: {args.data}")
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
        
    df = pd.read_csv(args.data)
    target = "Churn"
    
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found")
        sys.exit(1)
    
    logger.info(f"Data: {df.shape}, Target:\n{df[target].value_counts()}")

    # 3. Create versioned directory
    version_dir = create_version_dir(model_dir, config["model_artifacts"]["version_format"])

    # 4. Save baseline stats 
    exclude_cols = ['CustomerID', 'Tenure', 'Usage Frequency', 'Subscription Type', 'Contract Length', 'Churn']
    save_baseline_stats(df, version_dir, exclude_cols=exclude_cols)

    # 5. Prepare data
    df = preprocess(df=df)

    # --- Store PREPROCESS ---
    processed_data_path = root_dir / "data" / "preprocessed"
    processed_data_path.mkdir(parents=True, exist_ok=True) # create folder in case not exist
    
    output_file = processed_data_path / "train.parquet"
    df.to_csv(output_file, index=False)
    logger.info(f" Preprocessed data saved to: {output_file}")
    # ------------------------------

    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"], 
        random_state=config["training"]["random_state"], stratify=y
    )

    # 6. Build Pipeline
    pipeline = Pipeline([
        ("feat", FeatureEngineer()),
        ("prep", create_preprocessor()),
        ("model", RandomForestClassifier(random_state=config["training"]["random_state"]))
    ])

    # 7. Hyperparameter search
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

    # 8. MLflow + Training

    # FIX: avoid MLflow remote auth issue in CI
    if os.getenv("CI") == "true":
        mlflow.set_tracking_uri("file:./mlruns")

    try:
        mlflow.set_experiment("churn_model_enterprise")
    except Exception as e:
        logger.warning(f"MLflow setup warning: {e}")

    with mlflow.start_run(run_name=f"RF_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=args.n_iter,
            scoring=config["training"]["scoring"], cv=config["training"]["cv_folds"],
            n_jobs=-1, verbose=1, random_state=config["training"]["random_state"]
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                pass

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        try:
            mlflow.sklearn.log_model(best_model, "model")
        except Exception as e:
            logger.warning(f"MLflow log_model warning: {e}")

        # Save artifacts to version folder
        # 1. Model (joblib - hyperparams đã nằm sẵn trong này)
        model_path = version_dir / "model.pkl"
        joblib.dump(best_model, model_path)
        
        # 2. Metadata (NHẸ - chỉ những gì cần để debug/audit nhanh)
        metadata = {
            "version": version_dir.name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "feature_names": list(X.columns),
            "best_params": search.best_params_,
            "mlflow_run_id": run.info.run_id,
            "samples": {"train": len(X_train), "test": len(X_test)}
        }
        with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved artifacts: {version_dir}")

        # Update latest pointer (CRITICAL cho detect_drift.py)
        update_latest_pointer(model_dir, version_dir)

        # Summary
        logger.info("-" * 60)
        logger.info("TRAINING COMPLETE")
        for k, v in metrics.items():
            logger.info(f"{k.upper():12}: {v:.4f}" if isinstance(v, float) else f"{k.upper():12}: {v}")
        logger.info(f"Model: {model_path}")
        logger.info(f"MLflow: {run.info.run_id}")
        logger.info("-" * 60)

    # Cleanup old versions
    cleanup_old_versions(model_dir, config["model_artifacts"]["keep_versions"])
    
    logger.info("✅ Pipeline finished")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Churn Model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--model_dir", default=None, help="Model output directory")
    parser.add_argument("--config", default=None, help="Path to drift_config.yaml")
    parser.add_argument("--n_iter", type=int, default=10, help="Hyperparam search iterations")
    
    args = parser.parse_args()
    
    try:
        sys.exit(main(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)