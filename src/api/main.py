from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import json
import time
import threading

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.inference import clear_model_cache, load_model, predict_churn
from src.api.schemas import ChurnPredictionResponse, CustomerFeatures

# -----------------------------
# CONFIG
# -----------------------------
LOG_PATH = "logs/monitoring.json"

# -----------------------------
# METRICS
# -----------------------------
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total predictions served",
    ["label"]  
)
ERROR_COUNTER = Counter(
    "model_errors_total",
    "Total prediction errors"
)

LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Inference latency",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)

DRIFT_COUNT = Gauge(
    "feature_drift_count",
    "Number of drifted features"
)

PRED_DRIFT = Gauge(
    "prediction_drift",
    "Prediction drift flag"
)

CONFIDENCE = Gauge(
    "confidence_mean",
    "Average prediction confidence"
)

# -----------------------------
# BACKGROUND DRIFT UPDATER
# -----------------------------
def update_drift_metrics():
    while True:
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                continue

            latest = data[-1]
               
            DRIFT_COUNT.set(len(latest.get("drifted_features", [])))
            PRED_DRIFT.set(int(latest.get("prediction_drift", 0)))

            if "confidence_mean" in latest:
                CONFIDENCE.set(float(latest["confidence_mean"]))

        except Exception as e:
            print("Drift metrics error:", e)

        time.sleep(10)  # cập nhật mỗi 10s


# -----------------------------
# APP LIFECYCLE
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        app.state.model_load_error = None
    except Exception as e:
        app.state.model_load_error = str(e)

    # start background thread
    threading.Thread(target=update_drift_metrics, daemon=True).start()

    yield
    clear_model_cache()


app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade ML inference service with monitoring",
    lifespan=lifespan
)

# Auto HTTP metrics (KHÔNG override /metrics nữa)
Instrumentator().instrument(app).expose(app)

# -----------------------------
# HEALTH
# -----------------------------
@app.get("/health")
def health():
    err = getattr(app.state, "model_load_error", None)
    if err:
        return {"status": "degraded", "detail": err}
    return {"status": "ok"}


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(customer: CustomerFeatures):
    start = time.time()

    try:
        result = predict_churn(customer)

        # latency
        latency = time.time() - start
        LATENCY.observe(latency)

        # counter với label
        PREDICTION_COUNTER.labels(label=result.label).inc()

        # confidence (safe cast)
        if result.churn_probability is not None:
            CONFIDENCE.set(float(result.churn_probability))

        return result

    except Exception as e:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=400, detail=str(e))