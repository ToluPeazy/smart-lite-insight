"""Guards on the committed models/registry.json.

The registry is the source of truth for what the API serves, so drift between
`deployed`, `latest_version`, and the artefacts on disk is worth catching in
CI rather than on the Pi.
"""

import hashlib
import json
from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).parent.parent / "models" / "registry.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


class TestCommittedRegistry:
    def test_exactly_one_deployed_entry(self, registry):
        deployed = [m for m in registry["models"] if m.get("deployed")]

        assert len(deployed) == 1

    def test_latest_version_matches_the_deployed_entry(self, registry):
        """latest_version is bookkeeping, but it must not contradict reality."""
        deployed = next(m for m in registry["models"] if m.get("deployed"))

        assert registry["latest_version"] == deployed["version"]

    def test_deployed_entry_is_the_isolation_forest(self, registry):
        deployed = next(m for m in registry["models"] if m.get("deployed"))

        assert deployed["version"] == "1.0"
        assert deployed["model_name"] == "isolation_forest"

    def test_versions_are_unique(self, registry):
        versions = [m["version"] for m in registry["models"]]

        assert len(versions) == len(set(versions))

    def test_recorded_hashes_match_artefacts_present_on_disk(self, registry):
        """Vacuous in CI (the .joblib files are gitignored), real on a host."""
        models_dir = REGISTRY_PATH.parent
        mismatched = []

        for entry in registry["models"]:
            for file_field, hash_field in (
                ("model_file", "model_hash"),
                ("scaler_file", "scaler_hash"),
            ):
                artefact = models_dir / entry[file_field]
                recorded = entry.get(hash_field)
                if not artefact.is_file() or not recorded:
                    continue
                actual = hashlib.sha256(artefact.read_bytes()).hexdigest()
                if actual != recorded:
                    mismatched.append(entry[file_field])

        assert mismatched == []
