"""Tests for src/detect.py — model integrity verification.

Uses tiny synthetic artefacts (a 20-row Isolation Forest) rather than the real
2M-row models, so the integrity paths are covered without slowing CI down.
"""

import hashlib
import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.detect import AnomalyDetector, SecurityError

FEATURE_NAMES = ["global_active_power_kw", "voltage_v"]


@pytest.fixture
def tiny_models_dir(tmp_path):
    """Write a tiny model + scaler and a registry entry with correct hashes.

    Returns a (models_dir, registry_path) tuple.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    rng = np.random.default_rng(0)
    X_raw = rng.uniform(0, 5, (20, len(FEATURE_NAMES)))

    scaler = StandardScaler().fit(X_raw)
    model = IsolationForest(n_estimators=5, random_state=0).fit(scaler.transform(X_raw))

    model_file = "anomaly_isolation_forest_v1.0.joblib"
    scaler_file = "scaler_v1.0.joblib"
    joblib.dump(model, models_dir / model_file)
    joblib.dump(scaler, models_dir / scaler_file)

    registry = {
        "models": [
            {
                "version": "1.0",
                "model_name": "isolation_forest",
                "model_file": model_file,
                "scaler_file": scaler_file,
                "model_hash": hashlib.sha256(
                    (models_dir / model_file).read_bytes()
                ).hexdigest(),
                "scaler_hash": hashlib.sha256(
                    (models_dir / scaler_file).read_bytes()
                ).hexdigest(),
                "deployed": True,
                "feature_names": FEATURE_NAMES,
                "training_date": "2026-01-01T00:00:00",
                "n_training_samples": 20,
                "anomaly_rate": 0.05,
            }
        ],
        "latest_version": "1.0",
    }

    registry_path = models_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))

    return models_dir, registry_path


def rewrite_registry(registry_path, mutate):
    """Apply `mutate` to the deployed entry and write the registry back."""
    registry = json.loads(registry_path.read_text())
    mutate(registry["models"][0])
    registry_path.write_text(json.dumps(registry, indent=2))


class TestModelIntegrityCheck:
    def test_correct_hashes_load(self, tiny_models_dir):
        models_dir, _ = tiny_models_dir

        detector = AnomalyDetector(models_dir=str(models_dir))

        assert detector.metadata["version"] == "1.0"
        assert detector.feature_names == FEATURE_NAMES

    def test_missing_model_hash_raises(self, tiny_models_dir):
        models_dir, registry_path = tiny_models_dir
        rewrite_registry(registry_path, lambda e: e.pop("model_hash"))

        with pytest.raises(SecurityError, match="no hash recorded"):
            AnomalyDetector(models_dir=str(models_dir))

    def test_missing_scaler_hash_raises(self, tiny_models_dir):
        models_dir, registry_path = tiny_models_dir
        rewrite_registry(registry_path, lambda e: e.pop("scaler_hash"))

        with pytest.raises(SecurityError, match="no hash recorded"):
            AnomalyDetector(models_dir=str(models_dir))

    def test_wrong_model_hash_raises(self, tiny_models_dir):
        models_dir, registry_path = tiny_models_dir
        rewrite_registry(registry_path, lambda e: e.update({"model_hash": "0" * 64}))

        with pytest.raises(SecurityError, match="Model integrity check failed"):
            AnomalyDetector(models_dir=str(models_dir))

    def test_wrong_scaler_hash_raises(self, tiny_models_dir):
        models_dir, registry_path = tiny_models_dir
        rewrite_registry(registry_path, lambda e: e.update({"scaler_hash": "0" * 64}))

        with pytest.raises(SecurityError, match="Scaler integrity check failed"):
            AnomalyDetector(models_dir=str(models_dir))

    def test_tampered_model_file_raises(self, tiny_models_dir):
        """A swapped artefact is rejected even though the registry is intact."""
        models_dir, _ = tiny_models_dir
        artefact = models_dir / "anomaly_isolation_forest_v1.0.joblib"
        artefact.write_bytes(artefact.read_bytes() + b"tampered")

        with pytest.raises(SecurityError, match="Model integrity check failed"):
            AnomalyDetector(models_dir=str(models_dir))

    def test_missing_registry_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnomalyDetector(models_dir=str(tmp_path))


class TestModelSelection:
    def test_specific_version_also_verified(self, tiny_models_dir):
        """Fail-closed applies to an explicitly requested version too."""
        models_dir, registry_path = tiny_models_dir
        rewrite_registry(registry_path, lambda e: e.pop("model_hash"))

        with pytest.raises(SecurityError):
            AnomalyDetector(models_dir=str(models_dir), version="1.0")

    def test_unknown_version_raises(self, tiny_models_dir):
        models_dir, _ = tiny_models_dir

        with pytest.raises(ValueError, match="not found in registry"):
            AnomalyDetector(models_dir=str(models_dir), version="9.9")
