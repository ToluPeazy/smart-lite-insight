# Smart-Lite Insight

**Edge-AI Energy Monitoring & Anomaly Detection on Raspberry Pi 5**

An end-to-end machine learning pipeline that ingests household energy data, detects anomalies using Isolation Forest, serves predictions via FastAPI, and visualises results in a Streamlit dashboard — all running locally on a Raspberry Pi 5 (8 GB). Extended with a local LLM agent for natural-language data querying.

## Project Status

🚧 **Phase 3 — ML Models & API Serving** (up next)

✅ Phase 1 — Data Engineering Foundation
✅ Phase 2 — Exploration & Feature Engineering

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full roadmap.

## Architecture
```
[UCI Energy Dataset / Synthetic Replayer]
                ↓
       [Data Ingestion Service]
      (Python + SQLite + APScheduler)
                ↓
        [Feature Engineering]
       (Pandas + rolling/lag/cyclical)
                ↓
       [ML Training Pipeline]
      (Isolation Forest, LOF, Prophet)
                ↓
       [Model Registry + Versioning]
          (joblib + registry.json)
                ↓
       [FastAPI Inference API]
     (/anomaly/score, /forecast, /timeseries)
                ↓
       [Streamlit Dashboard]
      (Plotly charts + anomaly overlay)
                ↓
        [LLM Agent — Phase 5]
     (Ollama + tool-calling + chat UI)
```

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/ToluPeazy/smart-lite-insight.git
cd smart-lite-insight

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up environment
cp .env.example .env

# 5. Install pre-commit hooks
pre-commit install

# 6. Run tests
pytest

# 7. Generate synthetic demo data (no dataset download needed)
python -m seed.replayer --days 7
```

## Project Structure
```
smart-lite-insight/
├── data/
│   ├── raw/                    # Original datasets (gitignored)
│   └── processed/              # SQLite DB, labelled anomalies
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   └── 01_feature_engineering.ipynb
├── src/
│   ├── ingest.py               # Data ingestion + scheduling
│   ├── validate.py             # Schema validation
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Model training (Phase 3)
│   ├── detect.py               # Anomaly detection logic (Phase 3)
│   ├── serve.py                # FastAPI endpoints (Phase 3)
│   └── agent.py                # LLM agent (Phase 5)
├── seed/
│   └── replayer.py             # Synthetic data generator
├── dashboard/
│   └── app.py                  # Streamlit dashboard (Phase 4)
├── models/
│   ├── registry.json           # Model version tracking
│   └── *.joblib                # Trained models (gitignored)
├── tests/                      # Unit + integration tests
├── docs/
│   ├── schemas/                # Telemetry schema definitions
│   ├── PROJECT_PLAN.md         # Full project roadmap
│   └── SETUP.md                # Dev environment setup
├── pyproject.toml              # All Python config
└── .env.example
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, SQLite, APScheduler |
| ML | scikit-learn (Isolation Forest, LOF), Prophet, joblib |
| API | FastAPI, uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Agent | Ollama (Llama 3.1 8B), tool-calling |
| Quality | black, ruff, pytest, pre-commit |
| Deployment | Docker, Docker Compose |
| Hardware | Raspberry Pi 5 (8 GB) |

## Data Source

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — 2 million+ readings of household energy consumption at 1-minute intervals over ~4 years.

Alternatively, run `python -m seed.replayer` to generate synthetic data with realistic daily patterns and anomalies.

## License

MIT