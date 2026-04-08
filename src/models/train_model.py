import sys
from pathlib import Path

# Fix path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Thêm # noqa: E402 vào từng dòng import dưới đây
import argparse # noqa: E402
import json # noqa: E402
import shutil # noqa: E402
from datetime import datetime # noqa: E402

import joblib # noqa: E402
import mlflow # noqa: E402
import mlflow.sklearn # noqa: E402
import pandas as pd # noqa: E402
import yaml # noqa: E402
from sklearn.ensemble import RandomForestClassifier # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, train_test_split # noqa: E402
from sklearn.pipeline import Pipeline # noqa: E402

from src.features.engineer import FeatureEngineer, create_preprocessor # noqa: E402

def save_baseline_stats(df, save_dir):
    """
    Saves baseline statistics of training data for monitoring/drift detection.
    """
    stats = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "type": "numeric"
            }
        else:
            stats[col] = {
                "categories": df[col].astype(str).unique().tolist(),
                "type": "categorical"
            }

    baseline_path = Path(save_dir) / "baseline_stats.json"
    with open(baseline_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Baseline stats saved to {baseline_path}")


def create_version_dir(base_dir):
    """
    Creates a versioned directory based on timestamp to avoid overwriting.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = Path(base_dir) / f"run_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def update_latest_symlink(base_dir, version_dir):
    """Updates the 'latest' pointer and copies artifacts to the root models directory."""
    # FIX: Convert strings to Path objects
    base_dir = Path(base_dir)
    version_dir = Path(version_dir)
    
    latest_path = base_dir / "latest"
    latest_info_file = base_dir / "LATEST_VERSION.txt"

    # 1. Automatically copy model.pkl and baseline_stats.json to the root models/ folder
    # This ensures GitHub Actions finds them at a fixed path
    try:
        shutil.copy2(version_dir / "model.pkl", base_dir / "model.pkl")
        shutil.copy2(version_dir / "baseline_stats.json", base_dir / "baseline_stats.json")
        print(f"Successfully deployed latest artifacts to {base_dir}")
    except Exception as e:
        print(f"Failed to copy artifacts to root: {e}")

    # 2. Save reference path to text file
    try:
        with open(latest_info_file, "w") as f:
            f.write(str(version_dir.relative_to(base_dir.parent)))
    except Exception as e:
        print(f"Could not update LATEST_VERSION.txt: {e}")

    # 3. Attempt to create symlink (Works on Linux/Mac, or Windows with Admin)
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(version_dir, target_is_directory=True)
        print(f"Created symlink: {latest_path} -> {version_dir}")
    except Exception as e:
        print(f"Symlink skipped (Common on Windows without Admin): {e}")

def main(args):
    # 1. Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    target = "Churn"

    # 2. Create versioned directory
    version_dir = create_version_dir(args.model_dir)

    # 3. Save baseline stats (before split)
    save_baseline_stats(df, version_dir)

    # 4. Split data
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Pipeline
    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # 6. Hyperparameter search space
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10]
    }

    # 7. MLflow setup
    mlflow.set_experiment("churn_model_enterprise")

    with mlflow.start_run(run_name="RandomForest_Retrain_Flow") as run:
        print("Starting Hyperparameter Tuning...")

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # 8. Evaluation
        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted")
        }

        # 9. Log MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        # 10. Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(best_model, model_path)

        # 11. Save config YAML
        config = {
            "model": {
                "name": "churn_model",
                "target_variable": target,
                "best_model": "RandomForestClassifier",
                "parameters": search.best_params_,
                "metrics": metrics,
                "artifact_path": str(version_dir),
                "mlflow_run_id": run.info.run_id
            }
        }

        config_path = version_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # 12. Update latest pointer & Auto-copy to root
        update_latest_symlink(args.model_dir, version_dir)

        # 13. Summary
        print("-" * 40)
        print("TRAINING COMPLETE")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        print(f"Model saved to: {model_path}")
        print(f"Version directory: {version_dir}")
        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--model_dir", required=True, help="Base directory to store versioned models")
    parser.add_argument("--config_out", default="unused.yaml", help="(Deprecated) not used anymore")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of search iterations")

    args = parser.parse_args()
    main(args)