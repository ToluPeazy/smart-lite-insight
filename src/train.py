"""Model training for Smart-Lite Insight.

Trains anomaly detection models (Isolation Forest, Local Outlier Factor)
on the feature-engineered energy data, evaluates performance, and saves
models with semantic versioning to the local registry.

Usage:
    # Train default model (Isolation Forest)
    python -m src.train

    # Train with comparison
    python -m src.train --compare

    # Custom contamination rate
    python -m src.train --contamination 0.02
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.features import build_feature_matrix

# ── Defaults ──
DEFAULT_DB_PATH = "data/processed/energy.db"
DEFAULT_MODELS_DIR = "models"
DEFAULT_CONTAMINATION = 0.01  # Expected anomaly rate


# ── Data Loading ──


def load_training_data(
    db_path: str = DEFAULT_DB_PATH,
    site_id: str = "home-01",
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Load raw readings and build the feature matrix.

    Args:
        db_path: Path to SQLite database.
        site_id: Site to load data for.
        sample_frac: Optional fraction to subsample (for faster iteration).

    Returns:
        Feature-engineered DataFrame ready for training.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            global_active_power_kw,
            global_reactive_power_kw,
            voltage_v,
            global_intensity_a,
            sub_metering_1_wh,
            sub_metering_2_wh,
            sub_metering_3_wh
        FROM readings
        WHERE site_id = ?
        ORDER BY timestamp
        """,
        conn,
        params=(site_id,),
        parse_dates=["timestamp"],
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for site_id='{site_id}'")

    df = df.set_index("timestamp")

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).sort_index()

    logger.info(f"Loaded {len(df):,} raw readings for site '{site_id}'")

    df_features = build_feature_matrix(df, drop_na=True)
    logger.info(
        f"Feature matrix: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns"
    )

    return df_features


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """Get the list of numeric feature columns for training.

    Excludes non-numeric and categorical columns.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List of numeric column names.
    """
    exclude = {"time_of_use"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


# ── Model Training ──


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest model.

    Args:
        X: Scaled feature matrix (n_samples, n_features).
        contamination: Expected proportion of anomalies.
        random_state: Random seed.

    Returns:
        Fitted IsolationForest model.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def train_lof(
    X: np.ndarray,
    contamination: float = DEFAULT_CONTAMINATION,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    """Train a Local Outlier Factor model.

    Note: LOF with novelty=True can be used for prediction on new data.

    Args:
        X: Scaled feature matrix.
        contamination: Expected proportion of anomalies.
        n_neighbors: Number of neighbors for LOF.

    Returns:
        Fitted LocalOutlierFactor model.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X)
    return model


# ── Evaluation ──


def evaluate_model(
    model,
    X: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate an anomaly detection model.

    Since we don't have ground-truth labels for all data, we evaluate
    using the model's own predictions and anomaly scores to report
    detection statistics.

    Args:
        model: Fitted sklearn anomaly detection model.
        X: Scaled feature matrix.
        model_name: Name for logging.

    Returns:
        Dict with evaluation metrics.
    """
    # Predict: 1 = normal, -1 = anomaly (sklearn convention)
    predictions = model.predict(X)
    scores = model.decision_function(X)

    n_anomalies = (predictions == -1).sum()
    n_total = len(predictions)
    anomaly_rate = n_anomalies / n_total

    # Score statistics
    score_stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p01": float(np.percentile(scores, 1)),
        "p05": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }

    metrics = {
        "model_name": model_name,
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate": round(anomaly_rate, 4),
        "score_stats": score_stats,
    }

    logger.info(f"\n{'─' * 50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"  Total samples:  {n_total:,}")
    logger.info(f"  Anomalies:      {n_anomalies:,} ({anomaly_rate:.2%})")
    logger.info(
        f"  Score range:    [{score_stats['min']:.3f}, {score_stats['max']:.3f}]"
    )
    logger.info(
        f"  Score mean±std: {score_stats['mean']:.3f} ± {score_stats['std']:.3f}"
    )
    logger.info(f"{'─' * 50}")

    return metrics


# ── Model Registry ──


def get_next_version(models_dir: str = DEFAULT_MODELS_DIR) -> str:
    """Get the next semantic version from the registry.

    Args:
        models_dir: Path to models directory.

    Returns:
        Version string like "1.0", "2.0", etc.
    """
    registry_path = Path(models_dir) / "registry.json"

    if not registry_path.is_file():
        return "1.0"

    with open(registry_path) as f:
        registry = json.load(f)

    if not registry.get("models"):
        return "1.0"

    versions = [float(m["version"]) for m in registry["models"]]
    next_major = int(max(versions)) + 1
    return f"{next_major}.0"


def save_model(
    model,
    scaler: StandardScaler,
    feature_names: list[str],
    metrics: dict,
    model_name: str,
    models_dir: str = DEFAULT_MODELS_DIR,
) -> str:
    """Save model, scaler, and metadata to the registry.

    Args:
        model: Fitted sklearn model.
        scaler: Fitted StandardScaler.
        feature_names: List of feature column names.
        metrics: Evaluation metrics dict.
        model_name: Model type name (e.g. "isolation_forest").
        models_dir: Directory to save to.

    Returns:
        Version string of the saved model.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    version = get_next_version(models_dir)

    # Save model and scaler
    model_filename = f"anomaly_{model_name}_v{version}.joblib"
    scaler_filename = f"scaler_v{version}.joblib"

    joblib.dump(model, models_path / model_filename)
    joblib.dump(scaler, models_path / scaler_filename)

    # Update registry
    registry_path = models_path / "registry.json"
    if registry_path.is_file():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    entry = {
        "version": version,
        "model_name": model_name,
        "model_file": model_filename,
        "scaler_file": scaler_filename,
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "n_training_samples": metrics["n_total"],
        "anomaly_rate": metrics["anomaly_rate"],
        "score_stats": metrics["score_stats"],
    }

    registry["models"].append(entry)
    registry["last_updated"] = datetime.now().isoformat()
    registry["latest_version"] = version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Model saved: {model_filename} (v{version})")
    logger.info(f"Scaler saved: {scaler_filename}")
    logger.info(f"Registry updated: {registry_path}")

    return version


def load_latest_model(models_dir: str = DEFAULT_MODELS_DIR) -> tuple:
    """Load the latest model, scaler, and metadata from the registry.

    Args:
        models_dir: Path to models directory.

    Returns:
        Tuple of (model, scaler, registry_entry).
    """
    registry_path = Path(models_dir) / "registry.json"

    with open(registry_path) as f:
        registry = json.load(f)

    if not registry.get("models"):
        raise FileNotFoundError("No models in registry")

    latest = registry["models"][-1]
    models_path = Path(models_dir)

    model = joblib.load(models_path / latest["model_file"])
    scaler = joblib.load(models_path / latest["scaler_file"])

    logger.info(f"Loaded model: {latest['model_file']} (v{latest['version']})")

    return model, scaler, latest


# ── Main Training Pipeline ──


def train_pipeline(
    db_path: str = DEFAULT_DB_PATH,
    contamination: float = DEFAULT_CONTAMINATION,
    compare: bool = False,
    sample_frac: float | None = None,
) -> dict:
    """Run the full training pipeline.

    Args:
        db_path: Path to SQLite database.
        contamination: Expected anomaly rate.
        compare: If True, train both IF and LOF and compare.
        sample_frac: Optional subsample fraction for faster iteration.

    Returns:
        Dict with training results.
    """
    # Load and prepare data
    df = load_training_data(db_path=db_path, sample_frac=sample_frac)
    feature_names = get_numeric_features(df)
    X_raw = df[feature_names].values

    logger.info(f"Training features: {len(feature_names)}")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    logger.info(f"Scaled feature matrix: {X.shape}")

    results = {}

    # Train Isolation Forest
    logger.info("Training Isolation Forest...")
    if_model = train_isolation_forest(X, contamination=contamination)
    if_metrics = evaluate_model(if_model, X, "Isolation Forest")
    version = save_model(
        if_model,
        scaler,
        feature_names,
        if_metrics,
        model_name="isolation_forest",
    )
    results["isolation_forest"] = {
        "version": version,
        "metrics": if_metrics,
    }

    # Optionally train LOF for comparison
    if compare:
        logger.info("Training Local Outlier Factor...")
        lof_model = train_lof(X, contamination=contamination)
        lof_metrics = evaluate_model(lof_model, X, "Local Outlier Factor")
        lof_version = save_model(
            lof_model,
            scaler,
            feature_names,
            lof_metrics,
            model_name="lof",
        )
        results["lof"] = {
            "version": lof_version,
            "metrics": lof_metrics,
        }

        # Compare
        logger.info("\n" + "=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 50)
        for _name, r in results.items():
            m = r["metrics"]
            logger.info(
                f"  {m['model_name']}: {m['n_anomalies']:,} anomalies ({m['anomaly_rate']:.2%})"
            )
        logger.info("=" * 50)

    return results


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models for Smart-Lite Insight"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
        help=f"Expected anomaly rate (default: {DEFAULT_CONTAMINATION})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train both Isolation Forest and LOF, then compare",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Subsample fraction for faster iteration (e.g. 0.1 = 10%%)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    train_pipeline(
        db_path=args.db_path,
        contamination=args.contamination,
        compare=args.compare,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()
