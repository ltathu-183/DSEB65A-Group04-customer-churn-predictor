"""Load model file locally and run churn prediction (no Feast / Redis)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api.schemas import ChurnPredictionResponse, CustomerFeatures

# Raw columns matching train.csv after dropping CustomerID and Churn (stable order for DataFrame)
_RAW_COLUMN_MAP: list[tuple[str, str]] = [
    ("age", "Age"),
    ("gender", "Gender"),
    ("tenure", "Tenure"),
    ("usage_frequency", "Usage Frequency"),
    ("support_calls", "Support Calls"),
    ("payment_delay", "Payment Delay"),
    ("subscription_type", "Subscription Type"),
    ("contract_length", "Contract Length"),
    ("total_spend", "Total Spend"),
    ("last_interaction", "Last Interaction"),
]

def _default_model_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    model_dir = root / "models"
    joblib_path = (model_dir / "model.joblib").resolve()
    if joblib_path.is_file():
        return joblib_path
    return (model_dir / "model.pkl").resolve()


@lru_cache(maxsize=1)
def load_model() -> Any:
    """
    Load the local model once (cached) using joblib.

    Recommendation: The model should be a fitted end-to-end sklearn Pipeline 
    (preprocessing + model) to call .predict() directly on the raw data table defined above.
    """
    path = _default_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert JSON body to a single-row DataFrame with correct training column names."""
    row = {
        csv_name: getattr(customer, py_name)
        for py_name, csv_name in _RAW_COLUMN_MAP
    }
    return pd.DataFrame([row])


def predict_churn(customer: CustomerFeatures) -> ChurnPredictionResponse:
    """Return churn prediction from the local model (prefers .joblib)."""
    model = load_model()
    X = customer_to_dataframe(customer)

    pred = model.predict(X)
    churn_bool = bool(pred[0])

    proba: float | None = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # Assume positive class (churn) is at index 1 — standard for binary {0,1} and LabelEncoder
        if len(probs) >= 2:
            proba = float(probs[1])
        else:
            proba = float(probs[0])

    return ChurnPredictionResponse(
        churn=churn_bool,
        label="Churn" if churn_bool else "No Churn",
        churn_probability=proba,
    )


def clear_model_cache() -> None:
    """Utility for testing or reloading the model."""
    load_model.cache_clear()