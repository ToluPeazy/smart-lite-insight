"""Tests for scripts/backfill_model_hashes.py."""

import hashlib
import json

import pytest

from scripts.backfill_model_hashes import backfill_registry


def write_registry(models_dir, entry_overrides=None):
    """Create a registry with one entry and its two (fake) artefact files."""
    (models_dir / "anomaly_isolation_forest_v1.0.joblib").write_bytes(b"model-bytes")
    (models_dir / "scaler_v1.0.joblib").write_bytes(b"scaler-bytes")

    entry = {
        "version": "1.0",
        "model_name": "isolation_forest",
        "model_file": "anomaly_isolation_forest_v1.0.joblib",
        "scaler_file": "scaler_v1.0.joblib",
        "deployed": True,
    }
    entry.update(entry_overrides or {})

    registry_path = models_dir / "registry.json"
    registry_path.write_text(json.dumps({"models": [entry]}, indent=2))
    return registry_path


class TestBackfillRegistry:
    def test_records_missing_hashes(self, tmp_path):
        registry_path = write_registry(tmp_path)

        result = backfill_registry(tmp_path)

        assert sorted(result["updated"]) == ["1.0:model_hash", "1.0:scaler_hash"]
        entry = json.loads(registry_path.read_text())["models"][0]
        assert entry["model_hash"] == hashlib.sha256(b"model-bytes").hexdigest()
        assert entry["scaler_hash"] == hashlib.sha256(b"scaler-bytes").hexdigest()

    def test_is_idempotent(self, tmp_path):
        registry_path = write_registry(tmp_path)

        backfill_registry(tmp_path)
        first = registry_path.read_text()
        second_result = backfill_registry(tmp_path)

        assert second_result["updated"] == []
        assert sorted(second_result["unchanged"]) == [
            "1.0:model_hash",
            "1.0:scaler_hash",
        ]
        assert registry_path.read_text() == first

    def test_dry_run_does_not_write(self, tmp_path):
        registry_path = write_registry(tmp_path)
        before = registry_path.read_text()

        result = backfill_registry(tmp_path, dry_run=True)

        assert result["updated"]
        assert registry_path.read_text() == before

    def test_reports_mismatch_without_overwriting(self, tmp_path):
        registry_path = write_registry(tmp_path, {"model_hash": "0" * 64})

        result = backfill_registry(tmp_path)

        assert result["mismatched"] == ["1.0:model_hash"]
        entry = json.loads(registry_path.read_text())["models"][0]
        assert entry["model_hash"] == "0" * 64

    def test_skips_absent_artefacts(self, tmp_path):
        registry_path = write_registry(tmp_path)
        (tmp_path / "scaler_v1.0.joblib").unlink()

        result = backfill_registry(tmp_path)

        assert result["missing_files"] == ["scaler_v1.0.joblib"]
        assert result["updated"] == ["1.0:model_hash"]
        entry = json.loads(registry_path.read_text())["models"][0]
        assert "scaler_hash" not in entry

    def test_missing_registry_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            backfill_registry(tmp_path)

    def test_backfilled_registry_satisfies_detector(self, tmp_path):
        """End-to-end: hashes written here are the ones detect.py expects."""
        from src.detect import AnomalyDetector

        registry_path = write_registry(tmp_path)
        backfill_registry(tmp_path)
        entry = json.loads(registry_path.read_text())["models"][0]

        AnomalyDetector._verify_integrity(
            tmp_path / entry["model_file"], entry["model_hash"], "Model"
        )
        AnomalyDetector._verify_integrity(
            tmp_path / entry["scaler_file"], entry["scaler_hash"], "Scaler"
        )
