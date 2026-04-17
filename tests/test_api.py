import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app
from src.api.main import app
from src.api.schemas import ChurnPredictionResponse

# -----------------------------
# MOCKING STRATEGY
# -----------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_startup():
    """
    Globally mock heavy operations and background threads to prevent 
    the test suite from hanging or crashing due to missing files.
    """
    with patch("src.api.main.threading.Thread") as mock_thread, \
         patch("src.api.main.load_model") as mock_load:
        
        # Prevent background threads from actually starting 'while True' loops
        mock_thread.return_value.start = MagicMock()
        mock_load.return_value = None
        yield

@pytest.fixture
def client():
    """
    Provide a TestClient instance. 
    Using 'with' triggers FastAPI's lifespan (startup and shutdown events).
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_payload():
    """Return a valid customer feature dictionary matching schemas.py."""
    return {
        "age": 35,
        "gender": "Female",
        "tenure": 24,
        "usage_frequency": 15,
        "support_calls": 2,
        "payment_delay": 5,
        "subscription_type": "Premium",
        "contract_length": "Annual",
        "total_spend": 1250.50,
        "last_interaction": 10
    }

# -----------------------------
# TEST CASES
# -----------------------------

def test_health_check_ok(client):
    """Test the /health endpoint when the service is healthy."""
    app.state.model_load_error = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_degraded(client): # FIXED: Added 'client' parameter here
    """Test the /health endpoint when the model failed to load."""
    # Simulate a model loading error stored in app state
    app.state.model_load_error = "Mocked loading failure"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["detail"] == "Mocked loading failure"


@patch("src.api.main.predict_churn")
def test_predict_success(mock_predict_churn, client, valid_payload):
    """Test successful prediction with valid input data."""
    mock_predict_churn.return_value = ChurnPredictionResponse(
        churn=False,
        label="No Churn",
        churn_probability=0.15
    )

    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "No Churn"
    assert "churn_probability" in data


def test_predict_invalid_input(client, valid_payload):
    """Test that invalid values (like negative age) trigger a 422 error."""
    invalid_payload = valid_payload.copy()
    invalid_payload["age"] = -10  # Schema says age >= 0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


@patch("src.api.main.predict_churn")
def test_predict_internal_error(mock_predict_churn, client, valid_payload):
    """Test handling of unexpected exceptions during the prediction process."""
    mock_predict_churn.side_effect = Exception("Model Inference Crash")

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Model Inference Crash"


def test_metrics_endpoint(client):
    """Test that Prometheus metrics are correctly exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # FIXED: Check for metrics that were actually seen in your error logs
    # instead of OS-specific CPU metrics.
    assert "python_gc_objects_collected_total" in response.text
    assert "http_request_duration_seconds" in response.text