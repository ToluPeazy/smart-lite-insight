# Smart-Lite Insight — State Check (Phase R)

**Date:** 4 November 2026
**Base:** `origin/main` at `f9596c3` (confirmed unchanged; no later commits on any branch)
**Scope:** Audit of the post-Phase-A state, with tightly scoped follow-ups. No Phase 2 work, no retraining, no changes to `models/*.joblib`, scalers or registry hashes.
**Environment used for this check:** Linux container, bash, Python 3.13.18, scikit-learn 1.9.1, black 26.10.1, ruff 0.17.3. This is not the Windows/PowerShell development machine, and CI runs Python 3.11/3.12. Version-sensitive results are marked as such.

## Status key

- **OK**: verified, no action needed.
- **Gap**: verified problem, with an action proposed.
- **Unverified**: could not be checked here; the reason is given.

Effort: **S** < 30 min, **M** < half a day, **L** > half a day.

## 1. Baseline

| ID | Area | Finding | Evidence | Status | Proposed action | Effort |
|---|---|---|---|---|---|---|
| B1 | Git | `origin/main` still points at `f9596c3`. `git log --oneline f9596c3..origin/main` is empty, and `git log --all --not f9596c3` is empty, so no branch carries a commit after the base. There are no tags. | `git rev-parse origin/main` → `f9596c3383…`; branches: `main` and `origin/main` at `f9596c3`, plus two other remote branches (one at `f9596c3`, one at `eccd4e1`), neither with commits that are not on `main` | OK | None | — |
| B2 | Tests | 192 passed, 0 failed. Coverage **81.17%** against the 70% gate (last recorded value about 79%). Lowest-covered module: `src/train.py` at 58%. | `pytest --cov=src --cov=seed --cov-config=.coveragerc` → `192 passed`, `Total coverage: 81.17%` | OK | None | — |
| B3 | Lint | `ruff check .` → `All checks passed!`. `black --check .` → `34 files would be left unchanged`. The CI-scoped commands give the same result. | Command output | OK | None. The local black (26.x) is newer than the CI pin (`>=24.4`), so a CI run is still the authoritative check. | — |
| B4 | Hygiene | `.env`, `__pycache__/` and `*.egg-info/` are untracked and ignored. `compose/.env`: the `compose/` directory does not exist, and the bare `.env` pattern in `.gitignore` would match it at any depth. | `git ls-files \| grep -E '\.env$\|__pycache__\|egg-info'` → no output; `.gitignore:2,4,26` | OK | None | — |
| B6 | Hygiene | The test run leaves an untracked 0-byte `data/processed/energy.db`. R4 is one way to create it, but the file in the test run comes from `tests/test_serve.py::TestTimeSeriesEndpoint`, which queries the real database path unpatched (corrected during resolution; the first version of this report attributed it to R4 alone). | File appeared at the pytest timestamp; bisected by test file and test | Gap | Point the test at a temporary database with the real schema. Ignored by `.gitignore:30` in the meantime. | S |

## 2. Sourcery follow-ups from PR #2

Source: [ToluPeazy/smart-lite-insight#2](https://github.com/ToluPeazy/smart-lite-insight/pull/2), review `4953316361` (sourcery-ai, 17 Aug 2026), with two unresolved threads. The `gh` CLI is not available in this environment; the comments were read with the GitHub MCP integration (`pull_request_read` → `get_review_comments`, `get_reviews`, `get_comments`), which returns the same data.

| ID | Area | Finding | Evidence | Status | Proposed action | Effort |
|---|---|---|---|---|---|---|
| S1 | Backfill CLI | `--dry-run` is still present, and `--models-dir` is optional with the default `"models"`. Sourcery's suggested `main()` would have set `required=True` and dropped `--dry-run`. It was correctly **not** applied. | `scripts/backfill_model_hashes.py:103-112` | OK | Keep as is. | — |
| S2 | Backfill CLI | Sourcery's underlying point is valid: a missing registry still produces a raw traceback, because `main()` does not catch the `FileNotFoundError` raised at `:55`. | `scripts/backfill_model_hashes.py:115` (unguarded call) | Gap | Catch `FileNotFoundError` in `main()`, print `error: …` to stderr, and return 2. The flags stay unchanged. Diff: `task2-followups.patch`. | S |
| S3 | Backfill tests | Dry-run is covered **only at function level** (`backfill_registry(..., dry_run=True)`). The `--dry-run` flag and the default `--models-dir` are untested: no test calls `main()`. The `scripts/` directory is also outside `--cov`, so the gap does not show in the coverage numbers. | `tests/test_backfill_model_hashes.py:55-62` (function-level only); `grep -n "main" tests/test_backfill_model_hashes.py` → none; `Makefile:15` / `ci.yml` `--cov=src --cov=seed` | Gap | Add a `TestCli` class with four tests: `--dry-run` does not write; the default `models` directory is used when no flag is given; a missing registry exits with code 2 and no traceback; a mismatch exits with code 1. The missing-registry test fails on the current code and passes with S2. | S |
| S4 | Rate limits | Rate-limited endpoints: `POST /anomaly/score` with `@limiter.limit("30/minute")`, `GET /timeseries` with `"20/minute"`, and `GET /anomalies` with `"10/minute"`. `/health`, `/health/details` and `/model/info` are not limited. | `src/serve.py:325-332`, `:386-394`, `:502-507` | OK | None | — |
| S5 | Rate limits | The limiter is keyed on `CF-Connecting-IP` (stripped), falling back to the socket address. | `src/serve.py:80-102` (`Limiter(key_func=client_identifier)`); `tests/test_serve_ratelimit.py:41-66` | OK | None | — |
| S6 | Rate-limit tests | The current tests call the **real** paths and send limit + 1 requests: `/timeseries` 21 against 20, `/anomalies` 11 against 10, `/anomaly/score` 31 against 30. Each asserts no 429 before the limit and a 429 on the last request. Sourcery's test targeted `/score`, which returns **404**, with 11 requests against a limit of 30, so it would have passed without ever touching the limiter. The current suite does not have that defect. | `tests/test_serve_ratelimit.py:69-145`; probe: `POST /score` → `404` | OK | None | — |
| S7 | Rate-limit tests | The claim "malformed payload is rejected without consuming quota" is stated but not asserted (Sourcery's second point). It is true: in a probe, 40 malformed bodies all returned 422, and a valid request afterwards still got 30 non-429 responses before the 429. The same holds for requests without an API key (all 403). | `tests/test_serve_ratelimit.py:122-123`; `HANDOFF.md:85`; probe output: `malformed x40: [422]`, `valid after malformed: first30 [503] 31st 429`, `no key x40: [403]` | Gap | Add two assertions against `/anomaly/score`: one with limit + 5 malformed bodies, then a valid request is not 429; one with limit + 5 unauthenticated calls, then an authenticated call is not 429. | S |
| S8 | Rate-limit tests | `test_limit_is_per_client_ip` only asserts that the second IP is not 429. It would still pass if the limiter were removed entirely. | `tests/test_serve_ratelimit.py:84-100` | Gap | Also assert that the exhausted IP's next request **is** 429. | S |

## 3. LOF evaluation

| ID | Area | Finding | Evidence | Status | Proposed action | Effort |
|---|---|---|---|---|---|---|
| L1 | Method | Confirmed. LOF is fitted with `novelty=True`, and `evaluate_model` then calls `model.predict(X)` and `model.decision_function(X)` on the **training** matrix. scikit-learn's documentation advises against this: in novelty mode, each training point is returned as one of its own nearest neighbours, which biases it towards "inlier". | `src/train.py:161-167` (`novelty=True`), `:194-195` (`predict` / `decision_function` on `X`), `:420-421` (called with the training `X`) | Gap | Score training data with the fit-time `negative_outlier_factor_ - offset_`. This is identical to what `fit_predict` with `novelty=False` returns, and is now asserted in a test. Diff: `task3-lof-evaluation.patch`. | S |
| L2 | Effect | The registry figure is consistent with this bias. v2.0 (LOF) records `anomaly_rate: 0.0081` with `contamination=0.01`. The correct training-set flag rate is, by construction, ≈ the contamination (about 1%). A synthetic demonstration reproduces the direction of the effect: the biased path gives **0.88%**, the corrected path **1.00%**, and 24 of 200 correct flags are lost. | `models/registry.json` (v2.0 `anomaly_rate`); demo `lof_bias_demo.py`: replayer, 30 days, `--seed 42`, random subsample **n = 20,000**, `random_state=42`, 42 features, `n_neighbors=20` | Gap (mechanism) / Unverified (UCI magnitude) | The real v2.0 figure cannot be recomputed here: the UCI data and the `.joblib` files are not in the repository. Treat 0.81% as an artefact of the method, not a model property. | — |
| L3 | Effect | The 1.00% Isolation Forest rate (v1.0) is also set by construction (`contamination=0.01`), so it is not a detection result either. IF does not suffer from the self-neighbour bias. | `src/train.py:133-140`; v1.0 `anomaly_rate: 0.01` | Gap (interpretation) | Rename the figure in the docs to "training-set flag rate"; see D1–D4. | S |
| L4 | Trace | Places the biased v2.0 numbers reach: `models/registry.json` v2.0 (`anomaly_rate`, `score_stats`); `README.md:36` (diagram), `:133` (results table), `:248` ("comparable detection rates"). They do **not** appear in `HANDOFF.md` or `docs/results-report-2026-08.md`: HANDOFF mentions LOF only as the model that was mistakenly served, at `:63`, and the results report covers only IF v1.0. | `grep -rniE 'lof\|0\.81\|comparable'` over README, HANDOFF, docs, src | Gap | Docs: see D1–D4. Registry: leave the v2.0 entry untouched (hash rule); if you want it, a separate `evaluation_note` field is possible, pending your decision. | S |
| L5 | Agreement | The repository computes no IF–LOF agreement at all. On the synthetic demo, agreement between the two models' 1% flags is **low**: Cohen's κ = 0.056; Jaccard overlap = 0.034; 13 rows flagged by both. This is synthetic data only, so it says nothing about UCI. It does show that matching flag rates do not mean the models flag the same rows. | Demo output above | Unverified (UCI) | The diff adds `flag_agreement()` (κ, Jaccard, counts) and `monthly_flag_rates()` (month-to-month stability) to `--compare` output. Running it on UCI needs a documented subsample on the machine that holds the data, for example `--sample-frac` once T1 is fixed. | M |
| L6 | Timing | "LOF training time: 32 hours" probably includes evaluation. `evaluate_model` performs two extra full k-NN passes over 2M rows (`predict` and `decision_function`). The corrected path needs neither. | `src/train.py:194-195`, `:420-421` | Unverified | Re-time on the data machine if the figure is to be kept. Otherwise describe it as "about 32 h including in-sample scoring". | — |
| T1 | Method | `--sample-frac` samples **raw minute readings before** feature engineering. That breaks the 1-minute lags, rolling windows and differences: the features are computed across random gaps. Any subsampled run, including one used for an LOF re-evaluation, trains on corrupted features. | `src/train.py:86-91` | Gap | Sample after `build_feature_matrix`, or take a contiguous time slice. Not included in the diffs; it is a separate approval. | S |
| T2 | Registry | `save_model` sets `latest_version` to every newly trained version unconditionally. The next `python -m src.train` will break the invariant that `tests/test_registry.py:28-32` and `HANDOFF.md:59-61` depend on, which is the condition behind Phase S finding 7. | `src/train.py:327` | Gap | Update `latest_version` only when promoting a version (`deployed: true`). This changes no model artefacts. | S |
| T3 | Registry | `load_latest_model()` picks `models[-1]`, which is LOF v2.0, and calls `joblib.load` **without** the integrity check. It has no callers. It reproduces both the array-position incident and the fail-open load. | `src/train.py:339-364`; `grep -rn load_latest_model` → definition only | Gap | Delete it, or route it through `AnomalyDetector`. | S |

### Proposed LOF method (for approval)

1. **Training-set flags** (implemented in the diff): for LOF, use `negative_outlier_factor_ < offset_`, which is the same as `LocalOutlierFactor(novelty=False).fit_predict(X)`. For IF, keep `predict`.
2. **Hold-out alternative** (not implemented): fit on the earlier period, for example December 2006 to December 2009, and score the later period with `novelty=True`. This is the honest test of out-of-sample behaviour and it avoids the self-neighbour issue entirely. It requires T1 to be fixed, or a contiguous split.
3. **Reporting**: no ground-truth labels exist. Report only (a) training-set flag rates, labelled as fixed by `contamination`; (b) IF–LOF agreement (Cohen's κ, Jaccard, the count flagged by both); and (c) month-to-month flag-rate stability. No accuracy, precision, recall or "detection rate" wording.

### Claims that exceed the evidence (proposed rewording; not applied)

| ID | Location | Current text | Problem | Proposed text |
|---|---|---|---|---|
| D1 | `README.md:36` | `(33s training, 1%)     (32h training, 0.81%)` | The percentages are flag rates set by `contamination`; LOF's figure is biased (L1–L2). | `(33 s training)        (~32 h incl. scoring)` |
| D2 | `README.md:132` | `IF anomaly detection rate \| 1.00% (20,486 anomalies)` | Not a detection rate; there are no labels. | `IF training-set flag rate (contamination = 0.01) \| 1.00% (20,486 rows)` |
| D3 | `README.md:133` | `LOF anomaly detection rate \| 0.81% (16,662 anomalies)` | Biased in-sample method; the correct value is ≈1% by construction. | Remove the row, or: `LOF training-set flag rate \| pending re-evaluation (v2.0 figure used a biased in-sample method)` |
| D4 | `README.md:248` | "Isolation Forest massively outperforms LOF at scale — 33 seconds vs 32 hours on 2M+ readings, with comparable detection rates. LOF's pairwise distance computation makes it impractical for large datasets." | "Outperforms" holds only for cost; "comparable detection rates" is unsupported; "pairwise" is imprecise, since scikit-learn uses tree or brute-force k-NN. | "Isolation Forest is far cheaper than LOF at this scale: about 33 seconds against about 32 hours on 2M+ readings. Without ground-truth labels the two cannot be ranked on detection quality. Both flag about 1% of training rows because the contamination is set to 0.01, and they need not flag the same rows. LOF's cost is dominated by k-nearest-neighbour queries, which scale poorly to 2M rows in 42 dimensions." |
| D5 | `README.md:249` | "Feature engineering matters more than model choice" | No ablation exists in the repo. | "Feature engineering was the main design effort: 42 engineered features (temporal, lag, rolling, rate-of-change, sub-metering ratios). No ablation was run to separate its effect from the choice of model." |
| D6 | `README.md:250` | "the full stack (API + dashboard + model inference) runs comfortably on a Raspberry Pi 5" | Only the API process was measured (`docs/results-report-2026-08.md:39-41`); the dashboard and Ollama were not profiled. | "The API with model inference runs on a Raspberry Pi 5 (8 GB) at about 238 MB RSS and 65 ms p95 per 24-hour batch (see `docs/results-report-2026-08.md`). The dashboard and LLM were not profiled." |
| D7 | `README.md:251` | "making the system genuinely accessible" | No user evaluation. | "…intended to let non-technical users query ML outputs in natural language (not yet evaluated with users)." |
| D8 | `README.md:31` | "35+ features" | Inconsistent with "42" in `:130` and the registry (`feature_names`: 42). | "42 features" |
| D9 | `docs/results-report-2026-08.md:65` | "a silent mismatch there produces an empty result with no obvious cause" | Not empty: the API returns 404 and the base-load CLI raises (R3). | "…a mismatch makes the API return 404 `No data found for the given parameters` and the base-load CLI fail with `No readings found for site_id='home-01'`, and neither names the site mismatch as the cause." |
| D10 | `src/train.py:181-183` | "evaluate … to report detection statistics" | Same as D2. | Fixed in `task3-lof-evaluation.patch` (descriptive statistics; rate fixed by contamination). |

## 4. Residual items from earlier phases

| ID | Area | Finding | Evidence | Status | Proposed action | Effort |
|---|---|---|---|---|---|---|
| R1 | Phase S | **No security review report is committed.** None is in `docs/`, and none ever was. The only Phase S records are the summary table in HANDOFF and the PR #2 body. | `ls docs` → `phase0-audit-baseline.md`, `results-report-2026-08.md`, `schemas/`; `git log --all -- 'docs/*security*'` → no output; `HANDOFF.md:113-124` | Gap | Commit `docs/security-review-2026-08.md` from your original review notes. I can draft one from the PR #2 body and HANDOFF on request, but the review text is yours to supply. | M |
| R2 | Registry | `latest_version` (`"1.0"`) matches the only entry with `deployed: true` (v1.0, `isolation_forest`). v2.0 has no `deployed` key (absent rather than `false`), which the loader treats as not deployed. This is guarded by tests. | `models/registry.json:4,8,136`; `src/detect.py:81-87`; `tests/test_registry.py:23-38` | OK | Optional: set `"deployed": false` on v2.0 explicitly (this is not a hash field). Also see T2, which will break this on the next retrain. | S |
| R3 | Site IDs | The mismatch is still present: the replayer writes `home-synth` (`seed/replayer.py:36`), while the base-load CLI (`src/baseload.py:198`), the API (`src/serve.py` `site_id` defaults), `train.py:46` and the dashboard (`dashboard/app.py:133`, `index=0`) all default to `home-01`. On synthetic data this does **not** produce an empty profile. Instead, base-load raises `ValueError: No readings found for site_id='home-01' in the given range`, `/timeseries` returns 404 `No data found for the given parameters`, and `python -m src.train` raises. The replayer CLI has **no** `--site-id` flag. Only `docs/results-report-2026-08.md:65` warns about it; no test covers the cross-module default. | Reproduced: `python -m seed.replayer --days 2 --db-path <tmp>`, then `python -m src.baseload --db-path <tmp>` → `ValueError`; `/timeseries` → 404; with `--site-id home-synth` → 200 / valid profile | Gap | Either add `--site-id` to the replayer, or make the error messages name the available site IDs. Reword D9. Settle the identifier in the EcoHome contract. | S |
| R4 | API | `/health/details` reports `database_accessible: true` when the database **does not exist**, because `sqlite3.connect()` creates an empty file and `SELECT 1` succeeds. It also leaves that empty file behind (B6). | `src/serve.py:296-300`; probe after deleting the DB: `{'status': 'degraded', 'model_loaded': False, 'database_accessible': True}`, file size 0 | Gap | Open the database read-only (`sqlite3.connect(f"file:{path}?mode=ro", uri=True)`) and check that the `readings` table exists. Outside Phase R's stated scope; listed for scheduling. | S |
| R5 | HANDOFF | `SMARTLITE_API_KEY` is used consistently (`HANDOFF.md:87,93,99`; also `.env.example:36,40`, `docker-compose.yml:14,43`, `src/serve.py:126`). The bare `API_KEY` appears only at `:124`, where it describes the stale reference that was removed. | `grep -rnoE '\b[A-Z_]*API_KEY\b'` | OK | None | — |
| R6 | HANDOFF | The Finding 1 row still says "**The registry still needs the backfill run … until then the API starts with no model.**" That is stale: both entries carry hashes (committed in `f20f4eb`), and the Phase A report confirms an integrity-checked load on the Pi. | `HANDOFF.md:117`; `models/registry.json` `model_hash` / `scaler_hash` on v1.0 and v2.0; `docs/results-report-2026-08.md:32`; same at `HANDOFF.md:126` ("Operator actions that remain…") | Gap | Mark the row as done (backfill run in `f20f4eb`, verified on the Pi on 17 August 2026), and trim `:126` to the items that still remain. | S |
| R7 | HANDOFF | It references documents that are not in the repo: "See the PRD" (`:111`, the unified PRD lives outside the repo). `docs/PROJECT_PLAN.md` and `docs/SETUP.md` (`:124`) are named only as references that no longer exist, which is accurate. `compose/.env` (`:93`) refers to a directory that is no longer present. | `ls docs`; `HANDOFF.md:93,111,124` | Gap (minor) | At `:111`, say where the PRD lives ("unified PRD v1.2, kept outside this repository"). Leave `:124` as is. | S |
| R8 | HANDOFF | The slowapi gotcha is partly inaccurate. Position does not matter: `request` only has to be a parameter **named** `request`. A limited endpoint without it fails **at import** with `No "request" or "websocket" argument on function`, not "only under load". | `HANDOFF.md:81`; probe with slowapi 0.1.10: an endpoint without `request` raises at decoration, and `request` as the second parameter decorates fine | Gap (minor) | Reword: "A rate-limited endpoint must have a parameter named `request: Request` (slowapi fails at import otherwise). Don't name the body `request`." | S |
| R9 | Build | The file is named `dockerignore`, without the leading dot, so Docker ignores it, and `.git/`, `.venv/` and local databases are sent in the build context. The Dockerfile copies only named directories, so `.env` does not end up in the image. | `git ls-files dockerignore`; `Dockerfile` `COPY` lines; added in `26de8f5` | Gap (low) | `git mv dockerignore .dockerignore`. | S |

## 5. Can Phase R be closed?

**Not yet.** The code base is healthy: tests pass, coverage is 81%, the linters are clean, and the base matches `f9596c3`. What blocks closure is the following:

1. **The published LOF result is methodologically wrong.** The 0.81% figure comes from the self-neighbour bias, and "comparable detection rates" is not supported (L1–L4). The README needs the D1–D4 rewording, and the `train.py` fix needs approval.
2. **The Phase S security review report was never committed (R1).** The Phase S record is therefore incomplete.
3. **The Sourcery follow-ups (S2, S3, S7, S8) are pending approval.**
4. **HANDOFF has a stale Finding 1 status (R6).**

R3, R4, R8, R9 and T1–T3 are real but small. They can be closed in this phase or carried into the Phase 2 backlog. The UCI magnitude of the LOF bias and the IF–LOF agreement (L2, L5, L6) need a documented-subsample run on the machine that holds the data; they cannot be verified here.

## 6. Resolution

Everything blocking closure, and the smaller items listed above, was fixed in [ToluPeazy/smart-lite-insight#3](https://github.com/ToluPeazy/smart-lite-insight/pull/3), merged as `c713ca8`. Each fix that changes behaviour comes with a test that fails on the code at `f9596c3`. At merge: 227 tests pass, coverage 89.6%, black and ruff clean.

| ID | Resolution | Commit |
|---|---|---|
| S2, S3 | Backfill CLI exits with code 2 and a one-line error on a missing registry; CLI-level tests for `--dry-run`, the default directory, a missing registry and a mismatch. | `20c1170` |
| S7, S8 | Rate-limit tests assert that malformed and unauthenticated requests do not consume quota, and that the exhausted client receives 429. | `e4fad1a` |
| L1, L3, D10 | LOF training data scored from fit-time outlier factors; IF–LOF agreement and monthly flag rates reported by `--compare`. | `4e255a8` |
| D1–D4 | README describes IF and LOF figures as training-set flag rates; LOF figure marked for re-evaluation. | `39fb0ae` |
| D5–D9 | README and results-report claims limited to measured evidence. | `cd97092` |
| R4, B6 | `/health/details` opens the database read-only and requires the `readings` table; the timeseries test uses a temporary database. | `d7eda2e` |
| R3 | Replayer gains `--site-id`; the base-load error names the sites present in the database. | `30e2cde` |
| T1, T2, T3 | `--sample-frac` applied after feature engineering; training no longer changes the served model; unverified `load_latest_model()` removed. | `950e4db` |
| R9 | `dockerignore` renamed to `.dockerignore`. | `40cd99d` |
| R6, R7, R8 | HANDOFF: Finding 1 and operator actions brought up to date, PRD location stated, slowapi gotcha corrected. | `75d9813` |
| R1 | Phase S record committed as `docs/security-review-2026-08.md`, compiled from the committed remediation trail. This commit's message reads "record Phase R resolution in the state check"; the resolution itself was added afterwards, in this section. | `9e14b37` |
| PR review | `--compare` no longer fails after saving; `latest_version` is pinned to the served entry using `select_served_entry()`, which the detector shares; invalid `sample_frac` rejected; LOF training flags checked against the fitted matrix; monthly flag rates logged. | `4a77136` |
| PR review | HANDOFF promotion steps also set `latest_version` and restart the API; one training process per models directory. | `6deac3c` |

**Still open, and not blocking:**

- **L2, L5, L6.** The UCI-scale size of the LOF bias, IF–LOF agreement and the LOF timing need a run on the machine that holds the data, using a documented subsample (`python -m src.train --compare --sample-frac <f>`).
- **R2.** Optionally set `"deployed": false` explicitly on v2.0.
- **Unchanged API 404.** The API's 404 for an unknown `site_id` still does not name the available sites. Listing sites over the API is a contract change, so it is left for Phase 2.

**Phase R is closed.**
