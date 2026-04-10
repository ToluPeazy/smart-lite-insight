"""Tests for src/train.py and src/detect.py"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.detect import AnomalyDetector
from src.train import (
    evaluate_model,
    get_next_version,
    get_numeric_features,
    save_model,
    train_isolation_forest,
    train_lof,
)


@pytest.fixture
def sample_feature_df():
    """Create a sample feature-engineered DataFrame for training tests."""
    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-15", periods=n, freq="min")

    return pd.DataFrame(
        {
            "global_active_power_kw": rng.uniform(0.5, 5.0, n),
            "voltage_v": rng.normal(230, 3, n),
            "global_intensity_a": rng.uniform(2, 22, n),
            "hour": np.tile(np.arange(24), n // 24 + 1)[:n],
            "is_weekend": rng.integers(0, 2, n),
            "hour_sin": rng.uniform(-1, 1, n),
            "hour_cos": rng.uniform(-1, 1, n),
            "global_active_power_kw_lag_1m": rng.uniform(0.5, 5.0, n),
            "global_active_power_kw_roll_mean_1h": rng.uniform(1.0, 4.0, n),
            "global_active_power_kw_roll_std_1h": rng.uniform(0.1, 1.5, n),
            "global_active_power_kw_diff_1m": rng.normal(0, 0.5, n),
            "metered_ratio": rng.uniform(0.2, 0.8, n),
        },
        index=idx,
    )


@pytest.fixture
def scaled_data(sample_feature_df):
    """Return scaled feature matrix and scaler."""
    feature_names = get_numeric_features(sample_feature_df)
    X_raw = sample_feature_df[feature_names].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, scaler, feature_names


@pytest.fixture
def tmp_models_dir():
    """Create a temporary models directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty registry
        registry = {"models": [], "last_updated": None}
        with open(os.path.join(tmpdir, "registry.json"), "w") as f:
            json.dump(registry, f)
        yield tmpdir


class TestTrainIsolationForest:
    def test_returns_fitted_model(self, scaled_data):
        X, _, _ = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        assert hasattr(model, "predict")
        assert hasattr(model, "decision_function")

    def test_predictions_are_valid(self, scaled_data):
        X, _, _ = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        predictions = model.predict(X)
        assert set(np.unique(predictions)).issubset({-1, 1})

    def test_contamination_affects_anomaly_count(self, scaled_data):
        X, _, _ = scaled_data
        model_low = train_isolation_forest(X, contamination=0.01)
        model_high = train_isolation_forest(X, contamination=0.05)

        anomalies_low = (model_low.predict(X) == -1).sum()
        anomalies_high = (model_high.predict(X) == -1).sum()
        assert anomalies_high > anomalies_low


class TestTrainLOF:
    def test_returns_fitted_model(self, scaled_data):
        X, _, _ = scaled_data
        model = train_lof(X, contamination=0.01)
        assert hasattr(model, "predict")
        assert hasattr(model, "decision_function")

    def test_novelty_mode(self, scaled_data):
        X, _, _ = scaled_data
        model = train_lof(X, contamination=0.01)
        # Novelty=True allows predict on new data
        pred = model.predict(X[:5])
        assert len(pred) == 5


class TestEvaluateModel:
    def test_returns_expected_keys(self, scaled_data):
        X, _, _ = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "test_model")

        assert "model_name" in metrics
        assert "n_total" in metrics
        assert "n_anomalies" in metrics
        assert "anomaly_rate" in metrics
        assert "score_stats" in metrics

    def test_anomaly_count_reasonable(self, scaled_data):
        X, _, _ = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "test")

        assert metrics["n_anomalies"] > 0
        assert metrics["n_anomalies"] < metrics["n_total"]


class TestModelRegistry:
    def test_first_version(self, tmp_models_dir):
        version = get_next_version(tmp_models_dir)
        assert version == "1.0"

    def test_save_and_load(self, scaled_data, tmp_models_dir):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")

        version = save_model(
            model,
            scaler,
            feature_names,
            metrics,
            model_name="isolation_forest",
            models_dir=tmp_models_dir,
        )
        assert version == "1.0"

        # Check registry was updated
        with open(os.path.join(tmp_models_dir, "registry.json")) as f:
            registry = json.load(f)
        assert len(registry["models"]) == 1
        assert registry["models"][0]["version"] == "1.0"

    def test_version_increments(self, scaled_data, tmp_models_dir):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "test")

        v1 = save_model(model, scaler, feature_names, metrics, "if", tmp_models_dir)
        v2 = save_model(model, scaler, feature_names, metrics, "if", tmp_models_dir)

        assert v1 == "1.0"
        assert v2 == "2.0"


class TestAnomalyDetector:
    def test_init_loads_model(self, scaled_data, tmp_models_dir, sample_feature_df):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )

        detector = AnomalyDetector(models_dir=tmp_models_dir)
        assert detector.model is not None
        assert detector.scaler is not None

    def test_score_dataframe(self, scaled_data, tmp_models_dir, sample_feature_df):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )

        detector = AnomalyDetector(models_dir=tmp_models_dir)
        scored = detector.score_dataframe(sample_feature_df)

        assert "anomaly_score" in scored.columns
        assert "is_anomaly" in scored.columns
        assert "prediction" in scored.columns
        assert len(scored) == len(sample_feature_df)

    def test_get_anomalies(self, scaled_data, tmp_models_dir, sample_feature_df):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )

        detector = AnomalyDetector(models_dir=tmp_models_dir)
        scored = detector.score_dataframe(sample_feature_df)
        anomalies = detector.get_anomalies(scored)

        assert len(anomalies) > 0
        assert all(anomalies["is_anomaly"])

    def test_model_info(self, scaled_data, tmp_models_dir, sample_feature_df):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )

        detector = AnomalyDetector(models_dir=tmp_models_dir)
        info = detector.model_info

        assert info["model_name"] == "isolation_forest"
        assert info["version"] == "1.0"
        assert info["n_features"] == len(feature_names)
