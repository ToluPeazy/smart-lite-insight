"""Tests for src/serve.py"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from src.detect import AnomalyDetector
from src.serve import app, readings_to_dataframe
from src.train import (
    evaluate_model,
    save_model,
    train_isolation_forest,
)


#@pytest.fixture
#def client():
#    """Create a test client for the FastAPI app."""
#    return TestClient(app)

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    os.environ["SMARTLITE_API_KEY"] = "test-key-for-ci"
    return TestClient(app, headers={"X-API-Key": "test-key-for-ci"})


@pytest.fixture
def mock_detector(tmp_path):
    """Create a trained detector with a small synthetic dataset."""
    # Generate small training data
    n = 500
    rng = np.random.default_rng(42)
    feature_names = [
        "global_active_power_kw",
        "voltage_v",
        "global_intensity_a",
        "hour",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "global_active_power_kw_lag_1m",
        "global_active_power_kw_roll_mean_1h",
        "global_active_power_kw_roll_std_1h",
        "global_active_power_kw_diff_1m",
        "metered_ratio",
    ]

    X_raw = rng.uniform(0, 5, (n, len(feature_names)))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = train_isolation_forest(X, contamination=0.01)
    metrics = evaluate_model(model, X, "isolation_forest")

    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir)
    registry = {"models": []}
    with open(os.path.join(models_dir, "registry.json"), "w") as f:
        json.dump(registry, f)

    save_model(model, scaler, feature_names, metrics, "isolation_forest", models_dir)

    return AnomalyDetector(models_dir=models_dir)


@pytest.fixture
def sample_reading():
    """A single valid reading payload."""
    return {
        "timestamp": "2024-01-15T19:30:00",
        "global_active_power_kw": 4.216,
        "global_reactive_power_kw": 0.418,
        "voltage_v": 234.84,
        "global_intensity_a": 18.4,
        "sub_metering_1_wh": 0.0,
        "sub_metering_2_wh": 1.0,
        "sub_metering_3_wh": 17.0,
    }


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_accessible" in data


class TestModelInfoEndpoint:
    def test_returns_503_without_model(self, client):
        import src.serve as serve_module

        original = serve_module.detector
        serve_module.detector = None
        response = client.get("/model/info")
        assert response.status_code == 503
        serve_module.detector = original

    def test_returns_info_with_model(self, client, mock_detector):
        import src.serve as serve_module

        original = serve_module.detector
        serve_module.detector = mock_detector

        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "isolation_forest"
        assert "version" in data

        serve_module.detector = original


class TestReadingsToDataframe:
    def test_converts_readings(self, sample_reading):
        from src.serve import ReadingInput

        readings = [ReadingInput(**sample_reading)]
        df = readings_to_dataframe(readings)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 1
        assert "global_active_power_kw" in df.columns


class TestScoreEndpoint:
    def test_rejects_empty_batch(self, client):
        response = client.post("/anomaly/score", json={"readings": []})
        assert response.status_code == 422

    def test_validates_reading_fields(self, client):
        response = client.post(
            "/anomaly/score", json={"readings": [{"timestamp": "2024-01-15T19:30:00"}]}
        )
        assert response.status_code == 422


class TestTimeSeriesEndpoint:
    def test_returns_404_for_empty_data(self, client):
        response = client.get(
            "/timeseries",
            params={
                "start": "2099-01-01T00:00:00",
                "end": "2099-01-02T00:00:00",
                "site_id": "nonexistent",
            },
        )
        # Should return 404 or 503 depending on DB state
        assert response.status_code in [404, 503]


class TestAnomaliesEndpoint:
    def test_returns_503_without_model(self, client):
        import src.serve as serve_module

        original = serve_module.detector
        serve_module.detector = None

        response = client.get("/anomalies", params={"hours": 1})
        assert response.status_code == 503

        serve_module.detector = original


class TestOpenAPISchema:
    def test_docs_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Smart-Lite Insight API"

    def test_all_endpoints_documented(self, client):
        response = client.get("/openapi.json")
        schema = response.json()
        paths = list(schema["paths"].keys())
        assert "/health" in paths
        assert "/model/info" in paths
        assert "/anomaly/score" in paths
        assert "/timeseries" in paths
        assert "/anomalies" in paths
