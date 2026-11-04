"""Tests for src/train.py and src/detect.py"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.detect import AnomalyDetector
from src.train import (
    evaluate_model,
    flag_agreement,
    get_next_version,
    get_numeric_features,
    monthly_flag_rates,
    save_model,
    train_isolation_forest,
    train_lof,
    training_flags,
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


class TestTrainingFlags:
    def test_lof_training_flags_match_fit_predict(self, scaled_data):
        """Training-set flags equal what novelty=False fit_predict returns."""
        X, _, _ = scaled_data
        model = train_lof(X, contamination=0.05)
        expected = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=False
        ).fit_predict(X)

        predictions, _ = training_flags(model, X)

        np.testing.assert_array_equal(predictions, expected)

    def test_lof_training_flag_rate_matches_contamination(self, scaled_data):
        X, _, _ = scaled_data
        model = train_lof(X, contamination=0.05)

        predictions, _ = training_flags(model, X)

        assert (predictions == -1).mean() == pytest.approx(0.05, abs=1 / len(X))

    def test_lof_rejects_a_different_matrix(self, scaled_data):
        X, _, _ = scaled_data
        model = train_lof(X, contamination=0.05)

        with pytest.raises(ValueError):
            training_flags(model, X[:10])

    def test_isolation_forest_uses_predict(self, scaled_data):
        X, _, _ = scaled_data
        model = train_isolation_forest(X, contamination=0.05)

        predictions, scores = training_flags(model, X)

        np.testing.assert_array_equal(predictions, model.predict(X))
        np.testing.assert_allclose(scores, model.decision_function(X))


class TestFlagAgreement:
    def test_identical_flags(self):
        pred = np.array([1, 1, -1, 1, -1])

        result = flag_agreement(pred, pred)

        assert result["cohen_kappa"] == 1.0
        assert result["jaccard"] == 1.0
        assert result["flagged_both"] == 2

    def test_disjoint_flags(self):
        result = flag_agreement(np.array([-1, 1, 1, 1]), np.array([1, -1, 1, 1]))

        assert result["flagged_both"] == 0
        assert result["jaccard"] == 0.0
        assert result["cohen_kappa"] < 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            flag_agreement(np.array([1, -1]), np.array([1]))


class TestMonthlyFlagRates:
    def test_rates_per_month(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"])

        rates = monthly_flag_rates(idx, np.array([-1, 1, 1, 1]))

        assert rates == {"2024-01": 0.5, "2024-02": 0.0}


class TestLatestVersion:
    def test_second_model_does_not_change_latest_version(
        self, scaled_data, tmp_models_dir
    ):
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )
        save_model(model, scaler, feature_names, metrics, "lof", tmp_models_dir)

        with open(os.path.join(tmp_models_dir, "registry.json")) as f:
            registry = json.load(f)

        assert [m["version"] for m in registry["models"]] == ["1.0", "2.0"]
        assert registry["latest_version"] == "1.0"
        assert AnomalyDetector(models_dir=tmp_models_dir).metadata["version"] == "1.0"


class TestSampleFrac:
    def test_features_are_built_before_subsampling(self, tmp_path):
        """Sampled rows must equal the same rows of the full feature matrix."""
        from seed.replayer import replay_bulk
        from src.train import load_training_data

        db_path = str(tmp_path / "energy.db")
        replay_bulk(days=3, db_path=db_path, site_id="home-01", seed=7)

        full = load_training_data(db_path=db_path)
        sampled = load_training_data(db_path=db_path, sample_frac=0.2)

        assert 0 < len(sampled) < len(full)
        pd.testing.assert_frame_equal(sampled, full.loc[sampled.index])


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

    def test_prefers_deployed_flag_over_latest_appended(
        self, scaled_data, tmp_models_dir, sample_feature_df
    ):
        """A model marked deployed=true wins even if a newer version was
        appended afterwards -- this is the fix for the bug where the API
        silently served the last-trained model instead of the intended one.
        """
        X, scaler, feature_names = scaled_data
        model = train_isolation_forest(X, contamination=0.01)
        metrics = evaluate_model(model, X, "isolation_forest")
        save_model(
            model, scaler, feature_names, metrics, "isolation_forest", tmp_models_dir
        )
        save_model(model, scaler, feature_names, metrics, "lof", tmp_models_dir)

        registry_path = os.path.join(tmp_models_dir, "registry.json")
        with open(registry_path) as f:
            registry = json.load(f)
        assert registry["latest_version"] == "1.0"  # training does not promote
        registry["models"][0]["deployed"] = True
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        detector = AnomalyDetector(models_dir=tmp_models_dir)

        assert detector.metadata["version"] == "1.0"
