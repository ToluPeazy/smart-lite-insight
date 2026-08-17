"""Anomaly detection logic for Smart-Lite Insight.

Loads a trained model from the registry and scores new readings
for anomalies. Used by the FastAPI serving layer.

Usage:
    from src.detect import AnomalyDetector

    detector = AnomalyDetector()
    results = detector.score_dataframe(df_features)
"""

import hashlib
import hmac
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.features import build_feature_matrix
from src.train import DEFAULT_MODELS_DIR


class SecurityError(Exception):
    """Raised when a file integrity check fails."""


class AnomalyDetector:
    """Anomaly detection using a trained model from the registry.

    Loads the latest (or specified) model and scaler, then provides
    methods to score individual readings or DataFrames.
    """

    def __init__(
        self, models_dir: str = DEFAULT_MODELS_DIR, version: str | None = None
    ):
        """Initialise the detector by loading a model.

        Args:
            models_dir: Path to models directory.
            version: Specific version to load (None = latest).
        """
        self.models_dir = Path(models_dir)
        self.model, self.scaler, self.metadata = self._load_model(version)
        self.feature_names = self.metadata["feature_names"]

        logger.info(
            f"AnomalyDetector ready: {self.metadata['model_name']} "
            f"v{self.metadata['version']}"
        )

    def _load_model(self, version: str | None = None) -> tuple:
        """Load model, scaler, and metadata from registry.

        Model selection for version=None: prefer the entry explicitly marked
        "deployed": true so the choice survives future retraining and array
        reordering. Falls back to latest_version, then the last entry, for
        registries written before the deployed flag existed.
        """
        registry_path = self.models_dir / "registry.json"

        if not registry_path.is_file():
            raise FileNotFoundError(f"Registry not found: {registry_path}")

        with open(registry_path) as f:
            registry = json.load(f)

        if not registry.get("models"):
            raise FileNotFoundError("No models in registry")

        if version is not None:
            matches = [m for m in registry["models"] if m["version"] == version]
            if not matches:
                raise ValueError(f"Version {version} not found in registry")
            entry = matches[-1]
        else:
            deployed = [m for m in registry["models"] if m.get("deployed")]
            if deployed:
                entry = deployed[-1]
            else:
                latest_version = registry.get("latest_version")
                matches = [
                    m for m in registry["models"] if m["version"] == latest_version
                ]
                entry = matches[-1] if matches else registry["models"][-1]

        model_path = self.models_dir / entry["model_file"]
        scaler_path = self.models_dir / entry["scaler_file"]

        # Verify file integrity before deserializing. joblib.load() unpickles,
        # which executes code, so this check fails closed: an entry with no
        # recorded hash is rejected rather than loaded unverified.
        self._verify_integrity(model_path, entry.get("model_hash"), "Model")
        self._verify_integrity(scaler_path, entry.get("scaler_hash"), "Scaler")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler, entry

    @staticmethod
    def _verify_integrity(path: Path, expected_hash: str | None, label: str) -> None:
        """Check a file's SHA-256 against the registry, failing closed.

        Args:
            path: File about to be deserialised.
            expected_hash: Hash recorded in the registry, or None if absent.
            label: Human-readable name used in the error message.

        Raises:
            SecurityError: If the registry records no hash, or it doesn't match.
        """
        if not expected_hash:
            raise SecurityError(
                f"{label} integrity check failed for {path.name}: "
                "no hash recorded in the registry. Run "
                "'python scripts/backfill_model_hashes.py' to record one."
            )

        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual, expected_hash):
            raise SecurityError(f"{label} integrity check failed for {path.name}")

    def score(self, X_raw: np.ndarray) -> dict:
        """Score a single observation (raw feature values).

        Args:
            X_raw: 1D array of raw feature values (matching feature_names order).

        Returns:
            Dict with 'is_anomaly' (bool), 'anomaly_score' (float),
            and 'prediction' (-1 or 1).
        """
        X = self.scaler.transform(X_raw.reshape(1, -1))
        prediction = self.model.predict(X)[0]
        score = self.model.decision_function(X)[0]

        return {
            "is_anomaly": bool(prediction == -1),
            "anomaly_score": float(score),
            "prediction": int(prediction),
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of feature-engineered readings.

        Args:
            df: DataFrame with feature columns matching the trained model.

        Returns:
            Original DataFrame with added columns:
            'anomaly_score', 'is_anomaly', 'prediction'.
        """
        # Ensure we use the same features the model was trained on
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X_raw = df[self.feature_names].values
        X = self.scaler.transform(X_raw)

        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)

        result = df.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = predictions == -1
        result["prediction"] = predictions

        n_anomalies = result["is_anomaly"].sum()
        logger.info(
            f"Scored {len(result):,} readings: "
            f"{n_anomalies:,} anomalies ({n_anomalies / len(result):.2%})"
        )

        return result

    def score_raw_readings(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Score raw readings by first building features, then detecting.

        Convenience method that chains feature engineering and scoring.

        Args:
            df_raw: Raw readings DataFrame with DatetimeIndex.

        Returns:
            Feature-enriched DataFrame with anomaly scores.
        """
        df_features = build_feature_matrix(df_raw, drop_na=True)
        return self.score_dataframe(df_features)

    def get_anomalies(
        self,
        df_scored: pd.DataFrame,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Extract anomalous readings from a scored DataFrame.

        Args:
            df_scored: DataFrame with 'is_anomaly' and 'anomaly_score' columns.
            top_n: If set, return only the top N most anomalous readings.

        Returns:
            DataFrame of anomalous readings sorted by severity.
        """
        anomalies = df_scored[df_scored["is_anomaly"]].sort_values(
            "anomaly_score", ascending=True
        )

        if top_n is not None:
            anomalies = anomalies.head(top_n)

        return anomalies

    @property
    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.metadata["model_name"],
            "version": self.metadata["version"],
            "training_date": self.metadata["training_date"],
            "n_training_samples": self.metadata["n_training_samples"],
            "anomaly_rate": self.metadata["anomaly_rate"],
            "n_features": len(self.feature_names),
        }
