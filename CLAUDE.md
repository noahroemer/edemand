# CLAUDE.md — Multi-Region Probabilistic Demand Forecasting Platform

This file is the design doc and working agreement for this repo. Read it before making
changes. If a decision here conflicts with something you're about to do, raise it rather
than silently diverging.

## 1. What this project is

A deployed, monitored, day-ahead electricity demand forecasting system covering ~10 US
balancing authorities, producing calibrated probabilistic forecasts (not just point
forecasts), built on Databricks and served from a Dockerized API.

Scope in one sentence: ingest hourly demand + weather, engineer features in PySpark, fit
per-region and hierarchical Bayesian models, serve 24-hour-ahead forecasts with 80%
prediction intervals over HTTP, and monitor live forecast error, interval coverage, and
feature drift.

This is a portfolio project, built to be publicly readable and reproducible. Code quality,
README clarity, and honest evaluation matter as much as model accuracy.

## 2. Why it's built this way (resume gaps being filled)

The owner's existing experience already covers: PySpark/Databricks/Azure Spark in an
enterprise setting, LightGBM/XGBoost/Random Forest/Elastic Net/logistic regression,
walk-forward cross-validation, naive/ETS/linear-regression baselines, PyTorch for fairness
research, and classical hypothesis testing.

This project deliberately adds what's missing:

| Gap | How this project fills it |
| --- | --- |
| Deployment / MLOps | Dockerized FastAPI service, live public URL, CI/CD, model registry, drift-triggered retraining |
| Bayesian modeling | Hierarchical Bayesian model in PyMC with partial pooling |
| Uncertainty quantification | Posterior predictive intervals + conformal prediction, scored by coverage and CRPS |
| Owned, public, end-to-end Spark work | Multi-region Delta medallion pipeline the owner built and can show |
| Forecasting breadth | GAM / SARIMAX / reconciliation / foundation models |

**Hard constraint:** do not reach for models already on the resume. Specifically avoid
LightGBM, XGBoost, Random Forest, Elastic Net, and logistic regression as headline models.
Seasonal-naive is permitted only as a benchmark reference line, never reported as a
modeling result.

## 3. Environment and platform constraints

**Local dev (Windows):**

- Windows 10/11, PowerShell
- Python 3.12 in a venv at `.venv` (3.13 is installed on the machine but is too new for
  `databricks-connect` — do not use it)
- Activate with `.\.venv\Scripts\Activate.ps1`
- VS Code with the Databricks extension and Claude Code extension
- Git Bash installed (required by Claude Code on Windows)

**Databricks Free Edition — known limits:**

- Serverless compute only. No custom clusters, no cluster configs.
- Runs on Spark Connect. Use the DataFrame and SQL APIs only. Do not use RDD APIs,
  `sparkContext`, or custom Spark configs — they are unavailable.
- Outbound internet is restricted to a limited set of trusted domains unless unlocked via
  LinkedIn verification. This is the single biggest design variable (see Open Questions).
- Max 5 concurrent job tasks per account.
- Model Serving endpoints are limited; do not depend on them.
- Databricks Apps are only accessible to authenticated Databricks users, so an App cannot
  be the public demo.
- Accounts may be deleted after prolonged inactivity, so everything must be reproducible
  from this repo.
- Non-commercial use only.

**Implication:** the public serving layer is our own FastAPI container on Cloud Run /
Fly.io / Render — not Databricks.

**Decided (2026-10-05): Google Cloud Run, region `us-west1`.** Chosen over Fly.io and
Render because it scales to zero without the cold-start penalty that makes Render's free
tier a poor live demo, and because keyless OIDC auth from GitHub Actions is a stronger
answer to the §2 MLOps gap than a stored API token. Images live in Artifact Registry in the
same project and region — not GHCR, since a private repo's GHCR packages would require an
Artifact Registry remote repository as a bridge. Infrastructure setup is reproducible from
`docs/gcp-setup.md`.

## 4. Data

| Source | Content | Notes |
| --- | --- | --- |
| EIA API v2 | Hourly demand by balancing authority | Free API key. Paginate with `offset`; max ~5000 rows/request. Watch for gaps and revised values. |
| Open-Meteo | Historical archive + forecast weather | No key required. One or more grid points per region. |
| `holidays` package | US federal + state holidays | Major demand driver. |

Initial region set (10): CISO, ERCO, PJM, MISO, NYIS, ISNE, BPAT, SWPP, FPL, LDWP.

CAISO sub-areas for the hierarchical model: PGAE, SCE, SDGE, VEA.

**Non-negotiable data conventions:**

- Store everything in UTC. Convert to local time only when building calendar features.
  This is the number one source of bugs in this domain (DST transitions).
- Ingestion must be idempotent — use Delta `MERGE` upserts keyed on
  `(region, period_utc)`, never blind appends. EIA revises published values.
- Never silently forward-fill gaps. Flag them in a column and handle explicitly.

## 5. Architecture

```
EIA API v2  +  Open-Meteo
        |  ingestion (Databricks job, OR GitHub Actions - see Open Questions)
        v
+--------------------- Databricks ---------------------+
| bronze.demand_raw, bronze.weather_raw   (Delta)      |
| silver.demand, silver.weather           (PySpark)    |
| gold.features                           (PySpark)    |
|                                                      |
| train_fleet     GAM/SARIMAX per region               |
|                 via applyInPandas                    |
| train_caiso     PyMC hierarchical Bayesian           |
|                 + MinT reconciliation + conformal    |
| backtest        walk-forward -> gold.backtest_metrics |
| monitor         Spark SQL: rolling CRPS, coverage,   |
|                 PSI/KS drift, retrain trigger        |
|                                                      |
| MLflow tracking + Unity Catalog model registry       |
| All jobs defined as code in databricks/databricks.yml|
+------------------------------------------------------+
        |  artifact export (object storage or GitHub release)
        v
+--------------- Docker + Cloud -----------------------+
| FastAPI:  GET /health                                |
|           GET /predict?region=CISO                   |
|           GET /metrics                               |
| Serves fitted params / posterior draws - NO MCMC     |
| at request time                                      |
| Logs every prediction for later scoring              |
+------------------------------------------------------+
        ^
GitHub Actions: pytest, ruff, docker build, deploy
```

## 6. Modeling plan

| Role | Model | Runs where |
| --- | --- | --- |
| Benchmark reference | Seasonal-naive (same hour, last week) | Spark, all regions |
| Interpretable baseline | GAM (`pygam`) or SARIMAX with Fourier terms + weather exog | Spark `applyInPandas`, one fit per region |
| Primary model | Hierarchical Bayesian (PyMC): Fourier daily/weekly seasonality, spline temperature response, holiday effects, AR(1) errors, partial pooling across CAISO sub-areas | Single-node Databricks job |
| Coherence | MinT hierarchical reconciliation across CAISO sub-areas to total | Single-node |
| Calibration | Conformal prediction layered on every model | Single-node |
| Modern challenger (optional) | Chronos or TimesFM zero-shot; N-HiTS if time allows | Offline benchmark only |

**Rules:**

- MCMC does not distribute across Spark. Do not attempt to parallelize PyMC sampling with
  Spark. Fit single-node, log to MLflow.
- Heavy models (foundation models, deep learning) are offline benchmarks only. They do not
  go in the serving container.
- The serving container loads lightweight artifacts: fitted parameters or thinned posterior
  draws. Keep the image small and inference fast.

## 7. Evaluation

Primary metrics, reported per region and per regime:

- **CRPS** (continuous ranked probability score) — primary, since forecasts are distributional
- **Pinball loss** at the quantiles served
- **Empirical interval coverage** vs. nominal 80%
- **MAPE** — secondary, for comparability with conventional reporting
- **Diebold-Mariano tests** — to show whether model differences are statistically
  significant rather than noise

Regimes to break out: heat waves, cold snaps, federal holidays, weekday vs. weekend, and
by region.

Walk-forward CV only. No random K-fold on time series, ever. All models must be scored on
identical folds so comparisons are fair.

**Leakage warning:** backtests that use actual weather are optimistic, because at serving
time only forecast weather is available. Either train and evaluate against archived
weather forecasts, or state this limitation explicitly in the README. Do not quietly
report the optimistic number.

## 8. Repo layout

```
demand-forecast/
├── CLAUDE.md                  # this file
├── README.md                  # public-facing: architecture, results, limitations
├── databricks/
│   ├── 01_bronze_ingest.py
│   ├── 02_silver_clean.py     # PySpark
│   ├── 03_gold_features.py    # PySpark window functions
│   ├── 04_train_fleet.py      # applyInPandas: GAM/SARIMAX per region
│   ├── 05_train_caiso_bayes.py# PyMC + MinT + conformal
│   ├── 06_backtest.py         # walk-forward -> Delta metrics table
│   ├── 07_monitor.py          # Spark SQL rolling CRPS/coverage/drift
│   └── databricks.yml         # Asset Bundle - jobs as code
├── service/
│   ├── api.py                 # FastAPI app
│   ├── model_loader.py        # fetch registered artifact
│   ├── features.py            # serving-time feature construction
│   └── Dockerfile             # multi-stage
├── tests/
├── scripts/
│   ├── check_local_env.py     # day-0 environment doctor
│   └── smoke.py               # serving-contract assertions, stdlib only
├── docs/
│   └── gcp-setup.md           # one-time WIF + Artifact Registry + Cloud Run setup
├── requirements-service.txt   # hash-locked image manifest (uv pip compile)
├── .dockerignore
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .github/dependabot.yml
└── pyproject.toml
```

## 9. Key schemas

`gold.features` — one row per `(region, period_utc)`: `region`, `period_utc`, `demand_mw`,
lag features (48h, 168h), rolling means/stds, `temp_c`, `humidity`, `hour_local`, `dow`,
`month`, `is_holiday`, Fourier terms, `has_gap` flag.

**Decided (2026-10-10): demand lags are 48h and 168h, not 24h.** The forecast is issued at
13:00 UTC (06:00 PDT) for the next UTC day, so target hours run up to 34h past issue time.
A 24h lag for any target hour after about 13:00 UTC points at an hour that hasn't happened
yet at issue time. Backtests would look better than production could ever be. Rolling
statistics must also end at least 48h before the target hour. The 48h minimum assumes EIA
publishes each hour within about 14h. Measure the real delay once the daily job runs.

Prediction log — written on every served forecast:

```
issued_at        TIMESTAMP   -- when the forecast was made
target_time      TIMESTAMP   -- the hour being forecast (UTC)
region           STRING
model_version    STRING      -- MLflow model version
q10, q50, q90    DOUBLE
actual           DOUBLE      -- NULL at write time, backfilled by monitor job
```

Design this schema before writing the API. The entire monitoring story depends on being
able to join predictions to actuals later.

## 10. Serving contract

- Horizon: 24 hourly steps ahead
- Issued: once daily at a fixed time (06:00 America/Los_Angeles)
- `GET /health` — liveness
- `GET /predict?region=CISO` — 24 rows of `{target_time, q10, q50, q90, model_version}`
- `GET /metrics` — recent rolling error and coverage

## 11. Build order and milestones

Deploy a trivial baseline end to end early. Never be in a position of having a good model
and nothing shippable.

| Day | Work | Spark? |
| --- | --- | --- |
| 0 | Env setup. Run the two gating checks (§13). | — |
| 1 | Ingest 10 regions' demand + weather to bronze Delta | Light |
| 2 | Silver cleaning + MERGE incremental load. Dumb FastAPI in Docker, CI green, public deploy. | Yes |
| 3 | Gold feature table: window functions partitioned by region. EDA. | Yes |
| 4 | Seasonal-naive + GAM/SARIMAX per region via `applyInPandas`. Distributed walk-forward backtest to Delta metrics table. | Yes |
| 5 | CAISO hierarchical Bayesian model in PyMC. Diagnostics: R-hat, ESS, posterior predictive checks. | — |
| 6 | MinT reconciliation + conformal intervals. Compare Bayesian credible vs. conformal coverage. | — |
| 7 | Artifact export + registry-to-container handoff. Prediction logging live. Scheduled daily forecast job. | Light |
| 8 | Monitoring job: join predictions to actuals, rolling CRPS, coverage, PSI/KS drift, retrain trigger. | Yes |
| 9 | Diebold-Mariano tests, regime error analysis, one documented Spark performance fix. | Yes |
| 10 | Optional: Chronos/TimesFM zero-shot benchmark on a subset. | — |
| 11 | Asset Bundle, pytest, ruff, clean `docker compose up`. | — |
| 12 | README: architecture diagram, results table, methodology, limitations. | — |

Monitoring needs wall-clock time. Prediction logging must be live by day 7 so there is
real data to analyze by day 12.

**Deviation (2026-10-05): the CI/CD pipeline was built before day 1 ingestion.** The day-2
container and deploy work was pulled forward ahead of the data work. Rationale: §14 names
"deployment and CI debugging eats days" as the top risk and §11 already says to get
something shippable early, so this front-loads the mitigation. It also unblocks real work
while the EIA key and the §13 gating checks are still outstanding, since the serving
pipeline does not depend on either answer. Ingestion still must not be designed until §13
question 1 is resolved.

## 12. Conventions

- **Write PySpark first, not pandas.** The easiest way to lose the Spark learning value is
  to prototype in pandas and convert at the end. Transformations in `02_`, `03_`, `06_`,
  `07_` must be Spark DataFrame/SQL.
- Spark DataFrame and SQL APIs only — no RDDs, no `sparkContext`.
- Type hints on function signatures. `ruff` clean.
- `pytest` for feature transforms and API contract. Test the DST edge case and the
  gap-handling path specifically.
- Secrets via environment variables and GitHub Actions secrets. Never commit the EIA key
  or the Databricks PAT. `.databrickscfg` lives outside the repo.
- Commit messages: conventional commits (`feat:`, `fix:`, `chore:`).
- MLflow: `set_tracking_uri("databricks")`, `set_registry_uri("databricks-uc")`.
  Registered model names are three-level: `workspace.default.demand_forecast` (see §13 q3).

## 13. Open questions — resolve before building

These two answers change the design. Do not build past day 1 without them.

1. **Can Databricks serverless reach the EIA and Open-Meteo APIs?** Test in a notebook:
   `requests.get("https://api.eia.gov/v2/", timeout=10)`.
   - If yes, ingestion is a Databricks job.
   - If no, ingestion runs in GitHub Actions, writing Parquet to object storage that
     Databricks reads. Try LinkedIn verification first, which unlocks outbound internet
     access.
   - Note: during local dev via Databricks Connect, `requests` calls execute on the local
     machine, so this restriction may not surface until scheduling.
2. **Do `pymc`, `pygam`, `statsmodels`, and `properscoring` install on serverless?** Test
   with `%pip install` in a notebook. If PyMC will not install or is too slow, fall back to
   NumPyro, or reduce the spline basis / shorten the training history.
3. **Which Unity Catalog catalog?** **Decided (2026-10-06): `workspace`.** This workspace has
   no `main` catalog (`SHOW CATALOGS` returns only `samples`, `system`, `workspace`), so
   tables live in `workspace.{bronze,silver,gold}` and the registered model is
   `workspace.default.demand_forecast`. `.env.example` defaults `UC_CATALOG=workspace`.

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| Deployment and CI debugging eats days | Deploy a trivial baseline on day 2 |
| EIA pagination, gaps, DST timestamps | Store UTC, flag gaps, test the DST path |
| MCMC too slow on hourly data | Aggregate history, use NumPyro, reduce basis size |
| Weather leakage in backtests | Use archived forecasts, or disclose in README |
| Free Edition quotas / account deletion | Everything reproducible from this repo |
| Scope creep to all 65+ balancing authorities | Ship 10 regions solid first; expand only if time remains |

## 14a. Improvement backlog (after the baseline)

Ideas deliberately deferred so the baseline ships first. Each one is tried only after the
baseline exists, and kept only if the walk-forward backtest shows it helps. Record the
measured effect in the README either way.

- **Multi-point weather for large regions.** Each region currently uses one weather point
  (`ingest/regions.py`), which is crude for MISO, PJM, CISO, SWPP and ERCO. Upgrade to 3–5
  points weighted by metro-area population. Trigger: error by region is clearly worse for
  the large regions than for compact ones (LDWP, NYIS). CISO and LDWP currently share the
  same Los Angeles point.
- **Weight degree-hours, not temperatures.** When combining several points, average
  heating and cooling degree-hours (base 18 °C) rather than raw temperature. Demand responds
  to temperature in a U shape, so averaging a cold city and a mild one hides the heating load.
- **Solar radiation and cloud cover, especially for CISO.** Rooftop solar lowers measured
  grid demand on sunny afternoons. Open-Meteo provides `shortwave_radiation` and
  `cloud_cover`. Adding them is a change to `DEFAULT_VARIABLES` in `ingest/open_meteo.py`.
- **"Most recent available" same-hour lag.** One feature that uses demand from 24h back
  when that hour is already published at issue time, and 48h back otherwise. Fresher than a
  flat 48h lag for early target hours, still leak-free. Needs EIA's real publishing delay
  measured first, and must account for the issue time shifting between 13:00 and 14:00 UTC
  with daylight saving. Compare against `lag_48h` in the backtest.
- **Lagged temperature (e.g. 3h ago).** In CISO, July 2024, temperature peaks around
  15:00 local but demand peaks around 17:00–18:00, because buildings heat up slowly and
  people run AC after they get home. A lagged temperature feature would capture that delay.
- **Not doing: population and income as model features.** They don't vary hourly, the
  per-region models already absorb each region's level, and 10 regions is too few to learn
  their effect. Population is used only for weighting weather points.

## 15. Definition of done

- Public GitHub repo: README with architecture diagram, results table, honest limitations
  section.
- Live API URL returning 24-hour forecasts with 80% intervals.
- Databricks jobs defined as code in `databricks.yml` and running on schedule.
- Results section reporting CRPS and MAPE vs. seasonal-naive, empirical interval coverage,
  error by regime, and Diebold-Mariano test results.
- `docker compose up` works from a clean clone.

## 16. Working agreement with Claude Code

- Prefer small, reviewable commits over large rewrites.
- When a decision in this file would be violated, say so and ask rather than working
  around it.
- Don't invent benchmark numbers or fill results tables with placeholders that could be
  mistaken for real measurements. Mark unmeasured values as TBD.
- When touching the Databricks notebooks, remember they run on serverless Spark Connect —
  flag anything that needs a feature Free Edition doesn't have.
- Keep this file updated as decisions change. It is the source of truth for this project.
