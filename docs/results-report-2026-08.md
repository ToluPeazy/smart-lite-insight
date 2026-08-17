# Smart-Lite Insight — Phase A Deployment Results

**Target:** Raspberry Pi 5 (8 GB), 64-bit Raspberry Pi OS (arm64)
**Commit:** `62731d5` (`main`, post Phase S security remediation)
**Date:** 17 August 2026
**Author:** Tolu Babatunde

## 1. Purpose

This records the on-device deployment of Smart-Lite Insight on the Raspberry Pi 5 and the measured results against the Phase A acceptance criteria. Phase A is the gate before EcoHome integration (B1–B4): it establishes that the system runs on the target hardware with real, measured performance rather than estimates. This report is the committed evidence that closes that gate.

## 2. Environment

- Hardware: Raspberry Pi 5, 8 GB RAM.
- OS: 64-bit Raspberry Pi OS (arm64).
- Runtime: Docker with the Compose plugin; base image `python:3.11-slim` (multi-arch, arm64).
- Code: `main` at `62731d5`, which carries the Phase S security fixes.
- Model: Isolation Forest v1.0, trained on 2,048,561 samples (`training_date` 2026-02-24), 42 features. The `.joblib` artefacts were transferred to the Pi and verified against the registry hashes on load.
- Data: 7 days of synthetic readings from the seed replayer. 10,080 rows at one-minute resolution, 2026-08-10 to 2026-08-16, `site_id=home-synth`.

## 3. Acceptance criteria and results

| Criterion | Required | Result |
|---|---|---|
| Model version on-device | `/anomaly/score` and `/model/info` serve `model_version: "1.0"` | **Met.** `/model/info` returns version 1.0, Isolation Forest, 2,048,561 training samples. |
| On-device base-load profile | Schema-valid 48-slot profile generated on the Pi | **Met.** Generated via `src.baseload`; validates against `docs/schemas/baseload_v1.json`. |
| Measured performance | Real latency and resource metrics, not estimates | **Met.** See section 5. |
| Security controls active | Phase S fixes hold on real hardware | **Met.** See section 4. |

## 4. Security posture verified on-device

- **Model integrity.** The model loaded with the fail-closed integrity check active. Startup logged `AnomalyDetector ready: isolation_forest v1.0` with no `SecurityError`, confirming the SHA-256 of the transferred artefacts matched the hashes recorded in the registry.
- **Authentication over the public path.** `/model/info` returned the model metadata with a valid `X-API-Key`, and `HTTP 403 Forbidden` (`{"detail":"Invalid or missing API key"}`) with no key, confirming the endpoint is not publicly readable.
- **Rate limiting.** A burst above 30 requests per minute against `/anomaly/score` returned `HTTP 429`, confirming the limiter is active on the deployed instance.
- **Exposure.** Only the API (port 8000) was published through the Cloudflare Tunnel. The Streamlit dashboard (port 8501) was not exposed, and it sits behind the shared-secret gate (`SMARTLITE_DASHBOARD_PASSWORD`), which compose requires at startup.

## 5. Measured performance

- **Anomaly scoring latency** (in-process: feature engineering plus model inference), 24-hour batch of 1,440 readings, 50 iterations after warm-up: **p50 60 ms, p95 65 ms, max 80 ms.** The tight p50-to-p95 spread indicates stable per-batch cost with headroom on this hardware.
- **Memory.** The API process peaked at 243,232 kB (~238 MB) resident (`VmHWM`), model loaded and after the scoring runs, on an 8 GB device (roughly 3% of RAM). Read from `/proc/1/status` inside the container, because the Pi's kernel does not mount the memory cgroup controller, so `docker stats` reports zero.
- **Thermal.** 54.3°C under load, well below the throttle threshold.

Measurement note: latency here is the in-process compute cost. The end-to-end HTTP round-trip a remote client sees adds tunnel and serialisation overhead on top, and was not captured as a distribution because the rate limiter (working as intended) capped the burst probe.

## 6. Base-load profile

- **Method:** `submeter_subtraction`. The synthetic sub-metering channels were populated well enough for the real estimation path rather than the quantile fallback.
- **Result:** 48 half-hourly slots, mean base load 1.284 kW, over 2026-08-10 to 2026-08-16 for `site_id=home-synth`.
- **Validation:** passes `docs/schemas/baseload_v1.json` (`schema_version` `baseload-1.0`).

## 7. Public path verification

Published through an ephemeral Cloudflare quick tunnel (API only). From an external client:

- `GET /health` → `{"status":"ok"}`
- `GET /model/info` with key → v1.0 metadata
- `GET /model/info` without key → `HTTP 403`

This proves the API is reachable over the internet with authentication enforced across the tunnel, and the dashboard is not on any public route.

## 8. Caveats and limitations

- **Synthetic data.** The energy readings are seeded, not real. The deployment and the performance figures (latency, memory, thermal) are genuine and measured on the target hardware, but the base-load profile and anomaly outputs are not yet representative of a real home. Demonstrated real-world impact requires ingesting real data; this report evidences a working, measured deployment, which is a distinct and narrower claim than real-world impact.
- **Base-load numbers follow real data.** Once the UCI Individual Household Electric Power Consumption dataset is ingested, the base-load profile should be regenerated so the figures EcoHome consumes are representative.
- **Site identifier mismatch.** The seed replayer defaults to `site_id=home-synth` while the API and base-load default to `home-01`. This is harmless with the explicit flag, but it is a reconciliation item for the EcoHome data contract, since a silent mismatch there produces an empty result with no obvious cause.
- **Ephemeral tunnel.** Today's verification used a quick tunnel whose hostname rotates on restart. A named tunnel with a stable hostname (requiring a Cloudflare account and a domain) is a prerequisite for B1, where EcoHome needs a durable URL for `/baseload`.

## 9. What this unblocks

Phase A is met: the system runs on the Pi 5 with the security controls active and performance measured. This clears the gate to EcoHome integration (B1–B4). Immediate prerequisites for B1: a named tunnel with a stable hostname, ingestion of real UCI data so the base-load profile is representative, and agreement on the site identifier, base-load definition, and units in the EcoHome contract.
