# Smart-Lite Insight — Phase S Security Review (August 2026)

**Author:** Tolu Babatunde
**Reviewed code:** `main` at `8cd6bdd` (after the July 2026 hardening in PR #1)
**Remediated in:** PR #2, merged 17 August 2026 as `62731d5`
**Verified on device:** 17 August 2026, Raspberry Pi 5 (`docs/results-report-2026-08.md`)
**Record compiled:** 4 November 2026

## About this record

Phase S was the security review that gated deployment to the Raspberry Pi. The original review notes were not committed at the time. This record was compiled afterwards from what *was* committed: the PR #2 description, the per-finding commits (`security(finding-N): …`), the Phase S table in `HANDOFF.md`, and the on-device results report. Finding descriptions and severities are as recorded in PR #2. Where a fact could not be traced to one of those sources, this record says so rather than filling the gap.

## 1. Scope

**In scope:** the FastAPI service (`src/serve.py`), model loading and integrity (`src/detect.py`, `src/train.py`, `models/registry.json`), the Streamlit dashboard and in-process LLM agent (`dashboard/`, `src/agent.py`), runtime configuration (`docker-compose.yml`, `.env.example`), and declared dependencies (`pyproject.toml`).

**Out of scope** (as stated in PR #2): deployment and tunnel configuration, model retraining, and credential rotation, which are operator actions rather than code; and the base-load HTTP endpoint, multi-key authentication and a swappable model backend, which are later phases of the unified PRD.

**Method:** not recorded in the committed sources.

## 2. Earlier hardening (PR #1, July 2026)

PR #1 (merged 13 July 2026 as `8cd6bdd`) addressed an earlier set of eight issues. It forms the baseline this review started from:

- SHA-256 verification of `.joblib` files before deserialisation. It only ran when a hash was recorded, which is what Finding 1 below addresses.
- `/health` reduced to `{"status": "ok"}`; detail moved behind the API key at `/health/details`.
- Internal error detail stripped from HTTP responses and kept in server logs.
- An allow-list for the `resample` parameter.
- Batch size limit cut from 10,000 to 2,000 readings.
- CORS origins made configurable (`SMARTLITE_CORS_ORIGINS`) instead of a placeholder.
- LLM tool arguments filtered to the schema-defined parameters.
- A human-confirmation step for the `retrain_model` tool. Finding 2 below removes that tool entirely.

## 3. Findings

| # | Severity | Finding | Fix | Commit | Regression tests | Verified on device |
|---|---|---|---|---|---|---|
| 1 | High | `joblib.load` deserialised the production model with the integrity check silently skipped, because the registry carried no hash. | `src/detect.py` fails closed: a missing or mismatched hash raises `SecurityError`, compared in constant time. `src/train.py` records hashes for new models; `scripts/backfill_model_hashes.py` backfills existing entries. | `01f22e9`; hashes recorded in `f20f4eb` | `tests/test_detect.py`, `tests/test_backfill_model_hashes.py`, `tests/test_registry.py` | Yes. Startup logged `AnomalyDetector ready: isolation_forest v1.0` with no `SecurityError`. |
| 2 | High | The Streamlit dashboard bypassed the API key (in-process agent, direct database reads) and exposed a model-retrain tool. | Retrain tool removed from the agent. `dashboard/auth.py` adds a shared-secret gate (`SMARTLITE_DASHBOARD_PASSWORD`), enforced before anything renders and closed when the secret is unset; `docker-compose.yml` requires the variable. Exposure rules documented: API only. | `d53f86a` | `tests/test_agent.py`, `tests/test_dashboard_auth.py` | Partly. Port 8501 was not published through the tunnel, and compose required the secret. The gate itself was not exercised remotely. |
| 3 | Medium | The rate limiter keyed on the socket address, which behind the tunnel collapses every caller into one bucket; the heavy read endpoints were not limited. | Key on `CF-Connecting-IP`, falling back to the socket address. `/anomaly/score` 30/min, `/timeseries` 20/min, `/anomalies` 10/min. | `2735f04` | `tests/test_serve_ratelimit.py` | Yes. A burst above 30/min on `/anomaly/score` returned `HTTP 429`. |
| 4 | Low | A non-ASCII `X-API-Key` returned 500 instead of a clean authentication failure. | Compare UTF-8 bytes; return 403. | `6ea449d` | `tests/test_serve_auth.py` | Not specifically. A missing key returned `HTTP 403` over the tunnel; the non-ASCII case was covered by tests only. |
| 5 | Low | `reload=True` in the module entry point. | Auto-reload off by default; opt in with `SMARTLITE_RELOAD`. | `a5ab7f2` | `tests/test_serve_cli.py` | Not applicable. Compose starts uvicorn directly and does not call `main()`. |
| 6 | Low | `requests` imported but not declared as a dependency. | `requests` and `pydantic` declared in `pyproject.toml`. | `a2de01f` | `tests/test_packaging.py` | Not applicable. |
| 7 | Low | `latest_version` pointed at 2.0 while 1.0 was deployed. | Set to `"1.0"`. Model selection already preferred the `deployed` flag. | `813f9dc` | `tests/test_registry.py` | Yes. `/model/info` returned version 1.0. |
| 8 | Info | Stale documentation references raised in the review. | Verified as already resolved; recorded in `HANDOFF.md`. | `eccd4e1` | None | Not applicable. |

All High and Medium findings were fixed before deployment, along with the Low items.

## 4. Assumptions and residual risks

- **Cloudflare as the only ingress.** The rate limiter trusts `CF-Connecting-IP`, which Cloudflare sets and strips from client requests. If the API is ever reachable without Cloudflare in front, that header becomes caller-controlled and the limit can be evaded (`src/serve.py`, `client_identifier`).
- **Rate-limit state is in memory.** The limiter uses slowapi's default in-memory storage, with a single uvicorn process (`docker-compose.yml`). Counts reset when the API restarts, and would not be shared if more workers were added.
- **Requests rejected early do not count.** Malformed bodies (422) and requests without a valid key (403) are rejected before the limiter. This is deliberate, as both are cheap to reject, and it is covered by `tests/test_serve_ratelimit.py`.
- **The dashboard gate is a backstop, not authentication.** It is one shared secret with no accounts, lockout or rate limiting. Port 8501 must stay on the LAN or behind an authenticating proxy.
- **One shared API key.** Multi-key authentication is deferred to a later phase.
- **Ephemeral tunnel.** On-device verification used a quick tunnel whose hostname changes on restart. A named tunnel is a prerequisite for EcoHome integration (B1).

## 5. Follow-ups found later (Phase R, November 2026)

The Phase R state check (`docs/state-check-2026-11.md`) found two items related to this review, both fixed in Phase R:

- **An unverified model loader remained.** `load_latest_model()` in `src/train.py` loaded the last registry entry by array position with `joblib.load` and no integrity check. It had no callers, but it reproduced both the Finding 1 and Finding 7 failure modes. It has been removed.
- **`/health/details` misreported a missing database.** It reported `database_accessible: true` for a database that did not exist, and created an empty file as a side effect. It now opens the database read-only and checks for the `readings` table.
