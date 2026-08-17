# Smart-Lite Insight

**Edge-AI Energy Monitoring & Anomaly Detection on Raspberry Pi 5**

An end-to-end machine learning pipeline that ingests household energy data, detects anomalies using Isolation Forest, serves predictions via FastAPI, and visualises results in a Streamlit dashboard — all running locally on a Raspberry Pi 5 (8 GB). Includes a local LLM agent (Llama 3.1 8B via Ollama) for natural-language data querying with tool-calling capabilities.

## Project Status

✅ **All phases complete**

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Engineering Foundation | ✅ Complete |
| 2 | Exploration & Feature Engineering | ✅ Complete |
| 3 | ML Models & API Serving | ✅ Complete |
| 4 | Dashboard & Containerisation | ✅ Complete |
| 5 | LLM Agent | ✅ Complete |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    DATA LAYER                         │
│  [UCI Energy Dataset]    [Synthetic Replayer]         │
│         ↓                       ↓                    │
│  [Data Ingestion + Validation → SQLite]              │
│  (2M+ readings, schema-validated, 1-min intervals)   │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│                    ML LAYER                           │
│  [Feature Engineering Pipeline]                      │
│  (35+ features: lag, rolling, cyclical, sub-meter)   │
│         ↓                                            │
│  [Isolation Forest]     [Local Outlier Factor]       │
│  (33s training, 1%)     (32h training, 0.81%)        │
│         ↓                                            │
│  [Model Registry — semantic versioning + metadata]   │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│                    API LAYER                          │
│  [FastAPI — /health, /anomaly/score, /timeseries,    │
│   /anomalies, /model/info]                           │
│  (Pydantic validation, batch scoring, date ranges)   │
└──────────────────────────┬───────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────┐
│                  INTERFACE LAYER                      │
│  [Streamlit Dashboard]        [LLM Agent]            │
│  • Consumption chart          • Ollama + Llama 3.1   │
│  • Anomaly overlay            • 5 read-only tools    │
│  • Voltage stability          • Conversation memory  │
│  • Sub-metering breakdown     • Audit logging        │
│  • Model info sidebar         • Chat UI tab          │
└──────────────────────────────────────────────────────┘
```

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/ToluPeazy/smart-lite-insight.git
cd smart-lite-insight
cp .env.example .env       # set SMARTLITE_API_KEY and SMARTLITE_DASHBOARD_PASSWORD
docker compose up -d
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501 (asks for `SMARTLITE_DASHBOARD_PASSWORD`)

Both secrets are required — compose refuses to start without them. See [Security Notes](#security-notes) before exposing either port beyond the LAN.

### Option B: Local Development

```bash
git clone https://github.com/ToluPeazy/smart-lite-insight.git
cd smart-lite-insight

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# Install dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env

# Load data and train model
python -m src.ingest
python -m src.train

# Record model integrity hashes (only needed for models trained before
# the registry carried hashes — training records them automatically)
python scripts/backfill_model_hashes.py

# Start services
make dev
```

### Option C: Synthetic data (no dataset download)

```bash
python -m seed.replayer --days 7
python -m src.train
make dev
```

### LLM Agent Setup

```bash
# Install Ollama (https://ollama.com)
ollama pull llama3.1:8b

# Use via CLI
python -m src.agent

# Or via the "AI Chat" tab in the Streamlit dashboard
```

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 2,049,280 readings (Dec 2006 – Nov 2010) |
| Engineered features | 42 |
| Isolation Forest training time | 33 seconds |
| LOF training time | 32 hours |
| IF anomaly detection rate | 1.00% (20,486 anomalies) |
| LOF anomaly detection rate | 0.81% (16,662 anomalies) |
| API endpoints | 5 (health, score, timeseries, anomalies, model info) |
| LLM agent tools | 5 read-only (timeseries, anomalies, statistics, model info, date range) |

## Project Structure

```
smart-lite-insight/
├── data/
│   ├── raw/                        # UCI dataset (gitignored)
│   └── processed/                  # SQLite DB
├── notebooks/
│   ├── 00_data_exploration.ipynb   # EDA with 10 analysis sections
│   └── 01_feature_engineering.ipynb # Feature analysis and correlation
├── src/
│   ├── ingest.py                   # Data ingestion + scheduling
│   ├── validate.py                 # Schema validation
│   ├── features.py                 # Feature engineering (35+ features)
│   ├── train.py                    # Model training + registry
│   ├── detect.py                   # AnomalyDetector inference class
│   ├── serve.py                    # FastAPI REST endpoints
│   └── agent.py                    # LLM agent with tool-calling
├── seed/
│   └── replayer.py                 # 7-day synthetic data generator
├── dashboard/
│   ├── app.py                      # Streamlit dashboard (main)
│   ├── auth.py                     # Shared-secret gate for the dashboard
│   └── chat.py                     # AI chat tab
├── scripts/
│   └── backfill_model_hashes.py    # Record SHA-256 hashes in the registry
├── models/
│   ├── registry.json               # Model version tracking + integrity hashes
│   └── *.joblib                    # Trained models (gitignored)
├── tests/
│   ├── test_ingest.py
│   ├── test_validate.py
│   ├── test_features.py
│   ├── test_train.py
│   └── test_serve.py
├── docs/
│   └── schemas/telemetry_v1.json
├── HANDOFF.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, SQLite, APScheduler |
| ML | scikit-learn (Isolation Forest, LOF), joblib |
| API | FastAPI, uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Agent | Ollama, Llama 3.1 8B, tool-calling |
| Quality | black, ruff, pytest, pre-commit |
| Deployment | Docker, Docker Compose, Makefile |
| Hardware | Raspberry Pi 5 (8 GB) |

## API Endpoints

| Method | Endpoint | Description | Auth | Rate limit |
|--------|----------|-------------|------|------------|
| GET | `/health` | Liveness only (`{"status": "ok"}`) | — | — |
| GET | `/health/details` | Model + DB status | `X-API-Key` | — |
| GET | `/model/info` | Loaded model metadata | `X-API-Key` | — |
| POST | `/anomaly/score` | Score a batch of readings | `X-API-Key` | 30/min |
| GET | `/timeseries` | Retrieve data with optional anomaly overlay | `X-API-Key` | 20/min |
| GET | `/anomalies` | Find top anomalies in a time range | `X-API-Key` | 10/min |

Interactive Swagger docs at `http://localhost:8000/docs`.

Rate limits are per client IP and sized for a Pi 5 — `/timeseries` and `/anomalies` run feature engineering and model scoring over up to ~10k rows. The limiter identifies the caller by `CF-Connecting-IP` when present, falling back to the socket address, because behind `cloudflared` every request otherwise appears to come from the local tunnel endpoint and shares one bucket. **This assumes Cloudflare is the only ingress**: it sets that header and strips a client-supplied copy. Exposed any other way, the header is caller-controlled and the limit is evadable.

## LLM Agent Tools

| Tool | Type | Description |
|------|------|-------------|
| `get_timeseries` | Read | Fetch consumption data for a period |
| `get_anomalies` | Read | Find anomalous readings with severity scores |
| `get_statistics` | Read | Summary stats (mean, max, min, total kWh) |
| `get_model_info` | Read | Current model version and metrics |
| `get_date_range` | Read | Available data range in the database |

The tool set is read-only by design. The agent runs **in-process inside the dashboard**, which has no API-key check of its own, so it must not be able to trigger writes or long-running work. Retraining is an operator action: run `python -m src.train` on the host.

## Security Notes

### Exposing the stack

**Expose only the API (port 8000) through the Cloudflare Tunnel.** It is the only surface with authentication: every endpoint except `/health` requires `X-API-Key`, and the compute-heavy reads are rate limited.

**Do not tunnel the dashboard (port 8501).** It queries SQLite directly and runs the LLM agent in-process, so it never passes through the API key — anything that can reach 8501 can read all energy data. Keep it on the LAN, or put a real auth layer in front of it (Cloudflare Access, an authenticating reverse proxy, or a VPN).

As a backstop, the dashboard gates itself on `SMARTLITE_DASHBOARD_PASSWORD` and renders nothing until that secret is set and matched. It is one shared secret with no rate limiting or account model — a seatbelt against accidental exposure, not a substitute for the auth layer above.

### Agent tools

The LLM agent's tools are read-only. There is no retrain (or any other write) tool, because the agent is reachable from the dashboard, which is not authenticated by the API key. Retrain from the host with `python -m src.train`.

### Model integrity

Loading a model unpickles it, which executes code, so every registry entry carries a SHA-256 of its `.joblib` files and `AnomalyDetector` verifies them before loading. The check fails closed: an entry without a recorded hash is refused, not loaded unverified. Training records the hashes automatically; for older registries run `python scripts/backfill_model_hashes.py` on the machine holding the artefacts and commit the updated `models/registry.json`.

## Data Source

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — 2 million+ readings of household energy consumption at 1-minute intervals over ~4 years from a household in Sceaux, France.

Alternatively, run `python -m seed.replayer` to generate synthetic data with realistic daily patterns and injected anomalies.

## What I Learned

- **Isolation Forest massively outperforms LOF at scale** — 33 seconds vs 32 hours on 2M+ readings, with comparable detection rates. LOF's pairwise distance computation makes it impractical for large datasets.
- **Feature engineering matters more than model choice** — 42 engineered features (temporal, lag, rolling, rate-of-change, sub-metering ratios) gave the Isolation Forest rich signal to work with.
- **Edge deployment is viable** — the full stack (API + dashboard + model inference) runs comfortably on a Raspberry Pi 5 with 8GB RAM.
- **LLM tool-calling bridges the gap** — non-technical users can query complex ML outputs through natural language, making the system genuinely accessible.
- **Docker Compose simplifies everything** — one command to start the entire stack, reproducible across machines.

## License

MIT
