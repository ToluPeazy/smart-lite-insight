# HANDOFF

Working notes for anyone picking up Smart-Lite Insight after the original build. Read this before the code if you're new here — it covers the shape of the system, the state as of the Phase 0 audit, and the gotchas that have already cost time once.

## What this is

Edge-AI energy monitoring on a Raspberry Pi 5: ingest the UCI household power dataset (or a synthetic replayer) into SQLite, engineer ~42 time-series features, detect anomalies with a trained Isolation Forest, serve results over FastAPI, and visualise them in Streamlit with an optional local LLM (Ollama/Llama 3.1) chat agent on top.

```
UCI dataset / replayer → ingest.py → SQLite (data/processed/energy.db)
                                        ↓
                              features.py (42 features)
                                        ↓
                    train.py → models/*.joblib + models/registry.json
                                        ↓
                              detect.py (AnomalyDetector)
                                        ↓
        serve.py (FastAPI) ──┬── dashboard/app.py (Streamlit)
                              └── agent.py (Ollama tool-calling)
```

All five original build phases are complete. See `README.md` for the full architecture diagram, API table, and results.

## Module map

| File | Role |
|------|------|
| `src/ingest.py` | Loads the UCI CSV (or replayer output) into SQLite, with a scheduler for periodic ingestion |
| `src/validate.py` | Schema validation for incoming readings |
| `src/features.py` | Builds the feature matrix (lag, rolling, cyclical, sub-metering ratios) |
| `src/train.py` | Trains Isolation Forest / LOF, writes versioned entries to `models/registry.json` |
| `src/detect.py` | `AnomalyDetector` — loads a model from the registry and scores readings |
| `src/serve.py` | FastAPI app: `/health`, `/model/info`, `/anomaly/score`, `/timeseries`, `/anomalies` |
| `src/agent.py` | LLM tool-calling agent over Ollama (excluded from the coverage gate — no live-LLM tests yet) |
| `seed/replayer.py` | Synthetic data generator for running without the UCI download |
| `dashboard/app.py` | Streamlit dashboard (consumption chart, anomaly overlay, model info) |
| `dashboard/chat.py` | Streamlit tab wrapping `agent.py` |

## Model registry and the `deployed` flag

`models/registry.json` holds every trained model version as an array entry. **Do not rely on array position or `latest_version` alone to mean "what's live."** As of this audit, `AnomalyDetector._load_model()` selects the model to serve like this, in order:

1. The entry with `"deployed": true` (there should be exactly one).
2. If none is flagged, the entry matching `registry["latest_version"]`.
3. If neither is present (old-format registry), the last array entry — kept only for backward compatibility.

When you retrain and want to promote a new version to production, set `"deployed": true` on the new entry and remove it (or leave it false) on the old one — don't just append and bump `latest_version`, and don't reorder the array. The array order and `latest_version` are historical bookkeeping now, not the source of truth for what `serve.py` loads.

This exists because of a real incident: the API was silently serving v2.0 (LOF) via array-position fallback while every doc and the project narrative claimed v1.0 (Isolation Forest) was in production. Confirm this stays fixed by checking `GET /model/info` reports `"version": "1.0"` after any registry change.

## Model integrity check (fail closed)

`joblib.load()` unpickles, which executes arbitrary code, so `AnomalyDetector._load_model()` verifies the SHA-256 of both the model and the scaler against `model_hash` / `scaler_hash` in the registry entry **before** loading them.

**The check fails closed.** An entry with no recorded hash raises `SecurityError` — it is not loaded unverified. `src/train.py` records both hashes automatically when it saves a model, so anything trained after that change is fine; registries written earlier need a one-off backfill:

```bash
python scripts/backfill_model_hashes.py          # or --dry-run to preview
```

The script is idempotent — it only fills in hashes that are absent, skips entries whose `.joblib` files aren't on this machine, and exits non-zero on a mismatch rather than overwriting a recorded hash. The `.joblib` binaries are gitignored, so **run it on the machine that holds the artefacts (and on the Pi after transferring them), then commit the updated `models/registry.json`.** Until an entry has hashes, the API will start with no model and `/anomaly/*` returns 503 — that is the intended failure mode, not a bug to work around by softening the check.

If the check fires unexpectedly, the artefact on disk no longer matches what was trained. Re-copy it or retrain; do not "fix" it by deleting the hash from the registry.

## Known gotchas

**slowapi parameter ordering.** Any rate-limited endpoint (decorated with `@limiter.limit(...)`) must declare `request: Request` as its *first* parameter, with the request body named `payload` (or similarly, after `request`). slowapi's exception handler needs `request` on the function signature to work; if you add a new limited endpoint and put the body first, the limiter breaks in a way that's easy to miss locally and only shows up under load. See `score_readings(request: Request, body: BatchScoreRequest)` in `src/serve.py` for the pattern to copy.

**Read env vars inside the function, not at module import time.** `verify_api_key()` in `src/serve.py` calls `os.getenv("SMARTLITE_API_KEY")` inside the function body rather than caching it in a module-level constant at import time. Do the same for any new env-derived config that needs to reflect the current environment (tests monkeypatch env vars per-test; a module-level read would freeze the value at first import and ignore later changes).

**CORS is an explicit allow-list, not a wildcard.** `src/serve.py` sets `allow_origins` to a fixed list. Any new consumer (e.g. an EcoHome origin in Phase 3) needs to be added there explicitly — don't switch to `allow_origins=["*"]` as a shortcut.

**Secrets never go in tracked files.** `.env` and `compose/.env` are gitignored (and were untracked from history in the Phase 0 cleanup — rotate `SMARTLITE_API_KEY` if you ever suspect the old committed value leaked). `docker-compose.yml` reads `SMARTLITE_API_KEY` via `${SMARTLITE_API_KEY:?...}` substitution from the environment/`.env` — never hardcode a key literal into a compose file or Dockerfile.

## Dev workflow

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in SMARTLITE_API_KEY (see .env.example for how to generate one)
python -m src.ingest         # or: python -m seed.replayer --days 7
python -m src.train
make dev                     # API on :8000, dashboard on :8501
```

`make lint` runs `black --check` + `ruff check`; `make fmt` applies both. `make test` runs pytest with coverage (`--cov=src --cov=seed`, `src/agent.py` excluded via `.coveragerc`, gate is `fail_under = 70`). CI (`.github/workflows/ci.yml`) runs lint → test (3.11 and 3.12) → an ARM64 Docker build on pushes to `main`.

## Where things stand relative to the collaboration-readiness PRD

Phase 0 (this audit) fixed the model-loading bug, untracked `.env`/`compose/.env`/`egg-info`/`.pyc`, and committed this file. Phase 1 (base-load estimator) follows in `src/baseload.py`. See the PRD for the full phased plan (API contract hardening, external auth, swappable LLM backend, real data ingestion).
