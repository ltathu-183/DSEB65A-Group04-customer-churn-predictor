"""FastAPI: /health và /predict — chỉ đọc model local (.joblib/.pkl)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.inference import clear_model_cache, load_model, predict_churn
from src.api.schemas import ChurnPredictionResponse, CustomerFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up: kiểm tra load được model khi khởi động (fail fast)."""
    try:
        load_model()
    except FileNotFoundError as e:
        # Vẫn chạy app để /health báo trạng thái; /predict sẽ lỗi rõ ràng
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None
    yield
    clear_model_cache()


app = FastAPI(
    title="Churn Prediction API",
    description="Dự đoán churn từ model local (ưu tiên .joblib, fallback .pkl), không Feast / Redis.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Kiểm tra API và (nếu có) model đã load được."""
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
    Nhận đặc trưng khách hàng (JSON) và trả về dự đoán Churn.

    Body dùng tên field snake_case; bên trong được map sang tên cột CSV.
    """
    try:
        return predict_churn(customer)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Lỗi khi dự đoán (kiểm tra model local .joblib/.pkl và định dạng dữ liệu): {e}",
        ) from e
