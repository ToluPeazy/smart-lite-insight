"""FastAPI serving layer for Smart-Lite Insight.

Exposes endpoints for anomaly scoring, time-series retrieval,
and model metadata. Designed to run on the Raspberry Pi 5.

Usage:
    # Start the server
    python -m src.serve

    # Or with uvicorn directly
    uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.detect import AnomalyDetector
from src.features import build_feature_matrix
from src.train import DEFAULT_DB_PATH

# ── Global state ──
detector: AnomalyDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global detector
    try:
        detector = AnomalyDetector()
        logger.info("Model loaded successfully on startup")
    except FileNotFoundError:
        logger.warning("No trained model found — /anomaly endpoints will return 503")
        detector = None
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Smart-Lite Insight API",
    description="Energy anomaly detection API for Raspberry Pi 5",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-cloudflare-dashboard-url.trycloudflare.com",
        "http://localhost:8501",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key"],
)

# Rate limiter: 30 requests per minute per IP for scoring endpoint

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Custom Exception Handlers ──


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503, content={"detail": "Service temporarily unavailable"}
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")  # log internally
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# API_KEY = os.getenv("SMARTLITE_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# async def verify_api_key(key: str = Security(api_key_header)):
#    if not API_KEY or not secrets.compare_digest(key or "", API_KEY):
#        raise HTTPException(
#            status_code=status.HTTP_403_FORBIDDEN,
#            detail="Invalid or missing API key",
#        )


async def verify_api_key(key: str = Security(api_key_header)):
    api_key = os.getenv("SMARTLITE_API_KEY")
    if not api_key or not secrets.compare_digest(key or "", api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


# ── Request / Response Models ──


class ReadingInput(BaseModel):
    """Single energy reading for scoring."""

    timestamp: datetime = Field(..., description="Reading timestamp (ISO 8601)")
    global_active_power_kw: float = Field(..., ge=0, description="Active power in kW")
    global_reactive_power_kw: float = Field(
        ..., ge=0, description="Reactive power in kW"
    )
    voltage_v: float = Field(..., gt=0, description="Voltage in volts")
    global_intensity_a: float = Field(
        ..., ge=0, description="Current intensity in amps"
    )
    sub_metering_1_wh: float = Field(..., ge=0, description="Kitchen sub-meter (Wh)")
    sub_metering_2_wh: float = Field(..., ge=0, description="Laundry sub-meter (Wh)")
    sub_metering_3_wh: float = Field(
        ..., ge=0, description="Water heater/AC sub-meter (Wh)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp": "2024-01-15T19:30:00",
                    "global_active_power_kw": 4.216,
                    "global_reactive_power_kw": 0.418,
                    "voltage_v": 234.84,
                    "global_intensity_a": 18.4,
                    "sub_metering_1_wh": 0.0,
                    "sub_metering_2_wh": 1.0,
                    "sub_metering_3_wh": 17.0,
                }
            ]
        }
    }


class AnomalyResult(BaseModel):
    """Anomaly detection result for a single reading."""

    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    global_active_power_kw: float


class BatchScoreRequest(BaseModel):
    """Batch of readings for scoring."""

    readings: list[ReadingInput] = Field(..., min_length=1, max_length=10000)


class BatchScoreResponse(BaseModel):
    """Batch scoring results."""

    results: list[AnomalyResult]
    total: int
    anomaly_count: int
    anomaly_rate: float


class TimeSeriesPoint(BaseModel):
    """Single point in a time-series response."""

    timestamp: datetime
    global_active_power_kw: float
    anomaly_score: float | None = None
    is_anomaly: bool | None = None


class TimeSeriesResponse(BaseModel):
    """Time-series data with optional anomaly overlay."""

    data: list[TimeSeriesPoint]
    total_points: int
    start: datetime
    end: datetime
    anomaly_count: int | None = None


class ModelInfoResponse(BaseModel):
    """Model metadata."""

    model_name: str
    version: str
    training_date: str
    n_training_samples: int
    anomaly_rate: float
    n_features: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    database_accessible: bool


# ── Helper Functions ──


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}") from e


def require_detector() -> AnomalyDetector:
    """Ensure the detector is loaded."""
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.train' first.",
        )
    return detector


def readings_to_dataframe(readings: list[ReadingInput]) -> pd.DataFrame:
    """Convert a list of ReadingInput to a DataFrame with DatetimeIndex."""
    records = [r.model_dump() for r in readings]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


# ── Endpoints ──


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health, model status, and database connectivity."""
    db_ok = False
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if detector and db_ok else "degraded",
        model_loaded=detector is not None,
        database_accessible=db_ok,
    )


@app.get(
    "/model/info",
    dependencies=[Depends(verify_api_key)],
    response_model=ModelInfoResponse,
    tags=["Model"],
)
async def model_info():
    """Get metadata about the currently loaded model."""
    det = require_detector()
    info = det.model_info
    return ModelInfoResponse(**info)


@app.post(
    "/anomaly/score",
    dependencies=[Depends(verify_api_key)],
    response_model=BatchScoreResponse,
    tags=["Anomaly Detection"],
)
@limiter.limit("30/minute")
async def score_readings(request: Request, body: BatchScoreRequest):
    """Score a batch of readings for anomalies.

    Requires enough context for feature engineering (ideally 24h+ of
    sequential data). Short sequences may produce less reliable results
    due to rolling window warmup.
    """
    det = require_detector()

    df = readings_to_dataframe(body.readings)

    # Build features
    try:
        df_features = build_feature_matrix(df, drop_na=True)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature engineering failed: {e}. "
            f"Ensure readings are sequential with 1-minute intervals.",
        ) from e

    if df_features.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough data after feature engineering. "
            "Send at least 1440 sequential readings (24 hours) for reliable results.",
        )

    # Score
    scored = det.score_dataframe(df_features)

    results = []
    for ts, row in scored.iterrows():
        results.append(
            AnomalyResult(
                timestamp=ts,
                is_anomaly=bool(row["is_anomaly"]),
                anomaly_score=float(row["anomaly_score"]),
                global_active_power_kw=float(row["global_active_power_kw"]),
            )
        )

    anomaly_count = sum(1 for r in results if r.is_anomaly)

    return BatchScoreResponse(
        results=results,
        total=len(results),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(results), 4) if results else 0,
    )


@app.get(
    "/timeseries",
    dependencies=[Depends(verify_api_key)],
    response_model=TimeSeriesResponse,
    tags=["Time Series"],
)
async def get_timeseries(
    start: datetime = Query(None, description="Start timestamp (ISO 8601)"),
    end: datetime = Query(None, description="End timestamp (ISO 8601)"),
    hours: int = Query(
        24, ge=1, le=168, description="Hours of data (if start/end not given)"
    ),
    site_id: str = Query("home-01", description="Site identifier"),
    include_anomalies: bool = Query(False, description="Score data for anomalies"),
    resample: str = Query(None, description="Resample interval (e.g. '15min', '1h')"),
):
    """Retrieve time-series energy data with optional anomaly overlay.

    Either provide start/end timestamps, or use 'hours' to get the
    most recent N hours of data.
    """
    conn = get_db_connection()

    try:
        if start and end:
            query = """
                SELECT timestamp, global_active_power_kw, global_reactive_power_kw,
                       voltage_v, global_intensity_a,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(site_id, start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
        else:
            query = """
                SELECT timestamp, global_active_power_kw, global_reactive_power_kw,
                       voltage_v, global_intensity_a,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(site_id, hours * 60),
                parse_dates=["timestamp"],
            )
            df = df.sort_values("timestamp")
    except Exception as e:
        conn.close()
        raise HTTPException(
            status_code=503, detail=f"Database query failed: {e}"
        ) from e

    conn.close()

    if df.empty:
        raise HTTPException(
            status_code=404, detail="No data found for the given parameters"
        )

    df = df.set_index("timestamp")

    # Optional resampling
    if resample:
        df = df.resample(resample).mean().dropna()

    # Optional anomaly scoring
    anomaly_count = None
    scored_df = None
    if include_anomalies and detector is not None:
        try:
            df_features = build_feature_matrix(df, drop_na=True)
            if not df_features.empty:
                scored_df = detector.score_dataframe(df_features)
                anomaly_count = int(scored_df["is_anomaly"].sum())
        except Exception as e:
            logger.warning(f"Anomaly scoring failed: {e}")

    # Build response
    data = []
    for ts, row in df.iterrows():
        point = TimeSeriesPoint(
            timestamp=ts,
            global_active_power_kw=float(row["global_active_power_kw"]),
        )

        if scored_df is not None and ts in scored_df.index:
            point.anomaly_score = float(scored_df.loc[ts, "anomaly_score"])
            point.is_anomaly = bool(scored_df.loc[ts, "is_anomaly"])

        data.append(point)

    return TimeSeriesResponse(
        data=data,
        total_points=len(data),
        start=df.index.min(),
        end=df.index.max(),
        anomaly_count=anomaly_count,
    )


@app.get(
    "/anomalies", dependencies=[Depends(verify_api_key)], tags=["Anomaly Detection"]
)
async def get_anomalies(
    start: datetime = Query(None, description="Start timestamp"),
    end: datetime = Query(None, description="End timestamp"),
    hours: int = Query(24, ge=1, le=168, description="Hours to scan"),
    site_id: str = Query("home-01", description="Site identifier"),
    top_n: int = Query(50, ge=1, le=500, description="Max anomalies to return"),
):
    """Find the top anomalies in a time range.

    Returns the most severe anomalies sorted by anomaly score.
    """
    det = require_detector()
    conn = get_db_connection()

    try:
        if start and end:
            query = """
                SELECT timestamp, global_active_power_kw, global_reactive_power_kw,
                       voltage_v, global_intensity_a,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(site_id, start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
        else:
            query = """
                SELECT timestamp, global_active_power_kw, global_reactive_power_kw,
                       voltage_v, global_intensity_a,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(site_id, hours * 60),
                parse_dates=["timestamp"],
            )
            df = df.sort_values("timestamp")
    except Exception as e:
        conn.close()
        raise HTTPException(
            status_code=503, detail=f"Database query failed: {e}"
        ) from e

    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    df = df.set_index("timestamp")

    try:
        df_features = build_feature_matrix(df, drop_na=True)
        scored = det.score_dataframe(df_features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Scoring failed: {e}") from e

    anomalies = det.get_anomalies(scored, top_n=top_n)

    results = []
    for ts, row in anomalies.iterrows():
        results.append(
            {
                "timestamp": ts.isoformat(),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "global_active_power_kw": round(
                    float(row["global_active_power_kw"]), 3
                ),
                "voltage_v": round(float(row["voltage_v"]), 2),
            }
        )

    return {
        "anomalies": results,
        "total_found": len(results),
        "time_range": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
        },
    }


# ── CLI ──


def main():
    import uvicorn

    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
