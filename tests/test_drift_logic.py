import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from monitoring.detect_drift import DriftDetector, load_config

# -----------------------------
# FIXTURES
# -----------------------------

@pytest.fixture
def detector():
    """Initialize detector with default thresholds for testing."""
    return DriftDetector(p_val_threshold=0.05, psi_threshold=0.20)

@pytest.fixture
def stable_data():
    """Create two identical distributions (No Drift)."""
    np.random.seed(42)
    ref = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
    return ref, curr

@pytest.fixture
def drifted_data():
    """Create two different distributions (Significant Drift)."""
    np.random.seed(42)
    ref = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
    # Mean shift from 0 to 2.0 and scale shift
    curr = pd.DataFrame({"feat": np.random.normal(2.0, 1.5, 1000)})
    return ref, curr

# -----------------------------
# TEST CASES
# -----------------------------

def test_calculate_psi_identical(detector):
    """PSI should be near 0 for identical distributions."""
    data = np.random.normal(0, 1, 1000)
    psi = detector.calculate_psi(data, data)
    assert psi < 0.01


def test_calculate_psi_drifted(detector):
    """PSI should be high for different distributions."""
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(2, 1, 1000)
    psi = detector.calculate_psi(ref, curr)
    assert psi > 0.2  # Should exceed typical threshold


def test_integrity_check_fail(detector):
    """Test if integrity check catches high null values (>20%)."""
    df = pd.DataFrame({
        "Age": [20, 30, None, None, None], # 60% null
        "Gender": ["Male", "Female", "Male", "Male", "Male"]
    })
    # We need baseline_stats to check columns, let's mock it
    detector.baseline_stats = {"Age": {"type": "numeric"}}
    
    assert detector.check_integrity(df, "Test Data") is False


def test_numerical_drift_detection(detector, drifted_data):
    """Test if numerical drift is correctly identified."""
    ref, curr = drifted_data
    is_drifted = detector.detect_numerical_drift(ref["feat"], curr["feat"], "feat")
    assert is_drifted == True 



def test_categorical_drift_detection(detector):
    """Test Chi-square drift for categorical features."""
    ref = pd.Series(["Basic"] * 50 + ["Premium"] * 50)
    curr = pd.Series(["Basic"] * 10 + ["Premium"] * 90)
    
    is_drifted = detector.detect_categorical_drift(ref, curr, "Sub_Type")
    assert is_drifted == True

@patch("requests.post")
def test_trigger_retraining(mock_post, detector):
    """Test if the GitHub Action trigger sends the correct request."""
    # Mock environment variables
    with patch.dict("os.environ", {
        "GITHUB_REPOSITORY": "user/repo",
        "TOKENFORMLOPS": "fake_token"
    }):
        # Mock successful response (204 No Content)
        mock_post.return_value.status_code = 204
        
        detector.trigger_retraining("some/path.csv")
        
        # Verify API call
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert "dispatches" in args[0]
        assert kwargs["headers"]["Authorization"] == "Bearer fake_token"


def test_load_config_fallback():
    """Ensure config loader falls back to defaults if file is missing."""
    config = load_config("non_existent_path.yaml")
    assert "drift" in config
    assert config["drift"]["p_value_threshold"] == 0.05