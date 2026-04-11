"""FastAPI: /health and /predict — local model access only (.joblib/.pkl)."""
## uv run uvicorn src.api.main:app --reload 
# Run the command above to get the API_URL
# API_URL = http://127.0.0.1:8000
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.inference import clear_model_cache, load_model, predict_churn
from src.api.schemas import ChurnPredictionResponse, CustomerFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up: Verify model loading on startup (fail fast)."""
    try:
        load_model()
    except FileNotFoundError as e:
        # App continues to run so /health can report status; /predict will raise clear errors
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None
    yield
    clear_model_cache()


app = FastAPI(
    title="Churn Prediction API",
    description="Churn prediction using local models (prefers .joblib, falls back to .pkl), no Feast / Redis.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Check API status and verify if the model is loaded."""
    err = getattr(app.state, "model_load_error", None)
    if err:
        return {"status": "degraded", "model": "missing_or_unreadable", "detail": err}
    try:
        load_model()
    except Exception as e:
        return {"status": "degraded", "model": "error", "detail": str(e)}
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Receive customer features (JSON) and return Churn prediction.

    The request body uses snake_case fields, which are mapped internally to CSV column names.
    """
    try:
        return predict_churn(customer)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error (check local .joblib/.pkl model and data format): {e}",
        ) from e