# Phase 0 audit baseline — externally consumed response shapes

Captured verbatim from `src/serve.py` as it exists after the Phase 0 fixes, before any Phase 2 schema-versioning work. This is the baseline Phase 2 freezes and assigns a `schema_version` to — if a response shape below changes without Phase 2's schema-version-bump CI check in place, that's the drift Phase 2 exists to catch.

None of these responses currently carry a `schema_version` field. That is intentional and expected at this stage — Phase 2 adds it.

## `GET /health`

No auth. Backed by `HealthResponse`.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_accessible": true
}
```

## `GET /model/info`

Requires `X-API-Key`. Backed by `ModelInfoResponse`.

```json
{
  "model_name": "isolation_forest",
  "version": "1.0",
  "training_date": "2026-02-24T13:08:22.033424",
  "n_training_samples": 2048561,
  "anomaly_rate": 0.01,
  "n_features": 42
}
```

## `POST /anomaly/score`

Requires `X-API-Key`. Rate-limited (30/minute/IP). Backed by `BatchScoreResponse` (list of `AnomalyResult`).

```json
{
  "results": [
    {
      "timestamp": "2024-01-15T19:30:00",
      "is_anomaly": false,
      "anomaly_score": 0.1823,
      "global_active_power_kw": 4.216
    }
  ],
  "total": 1,
  "anomaly_count": 0,
  "anomaly_rate": 0.0,
  "model_version": "1.0"
}
```

`model_version` was added as part of the Phase 0 model-loading fix — the response previously carried no indication of which model produced the scores.

## `GET /timeseries`

Requires `X-API-Key`. Backed by `TimeSeriesResponse` (list of `TimeSeriesPoint`).

```json
{
  "data": [
    {
      "timestamp": "2024-01-15T19:30:00",
      "global_active_power_kw": 4.216,
      "anomaly_score": 0.1823,
      "is_anomaly": false
    }
  ],
  "total_points": 1,
  "start": "2024-01-15T19:30:00",
  "end": "2024-01-15T19:30:00",
  "anomaly_count": 0
}
```

`anomaly_score` / `is_anomaly` on each point, and the top-level `anomaly_count`, are only populated when `include_anomalies=true` is passed; otherwise they are `null`.

## `GET /anomalies`

Requires `X-API-Key`. **Not backed by a Pydantic `response_model`** — returns a plain dict, so today nothing validates its shape. This is the endpoint most likely to drift silently; Phase 2 should give it a proper response model alongside the schema version.

```json
{
  "anomalies": [
    {
      "timestamp": "2024-01-15T19:30:00",
      "anomaly_score": 0.1823,
      "global_active_power_kw": 4.216,
      "voltage_v": 234.84
    }
  ],
  "total_found": 1,
  "time_range": {
    "start": "2024-01-15T19:30:00",
    "end": "2024-01-15T19:30:00"
  }
}
```

## Not yet present

`GET /baseload` and `POST /anomaly/batch` don't exist yet — Phase 1 ships the base-load estimator as a function/CLI only (no HTTP surface); Phase 2 adds both endpoints per the PRD.
