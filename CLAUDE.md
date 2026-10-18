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

**LDWP data quality (decided 2026-10-12): keep all 10 regions, report LDWP separately.**
LDWP's published demand is materially worse than the other nine. Seasonal-naive scores
31.5% MAPE there against 6–11% everywhere else, and 553 hours of 2024 (6.3% of the year)
fall below 20% of its own median, including stretches pinned near 175–250 MW against a
2,503 MW median. That is bad data, not a hard region.

It stays in the region set, but:

- Any cross-region average must be reported both with and without LDWP, or LDWP alone will
  dominate it and the results table will read as incompetent rather than honest.
- The README limitations section names LDWP explicitly as a data-quality caveat.
- Do not tune modeling choices to improve LDWP. Its error is dominated by source data, so
  gains there measure nothing.

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

**Backtest design (decided 2026-10-15), implemented in `databricks/06_backtest.py`:**

- **21 monthly folds, expanding window.** Test months 2025-01 → 2026-09. Each fold trains on
  everything from 2023-01-01 up to 48h before its test month starts (the same 48h embargo as
  the lags), and refits once. Fold definitions live in `workspace.gold.backtest_folds` and
  every model reads them from there.
- **Every model outputs 19 quantiles** per hour (τ = 0.05, 0.10, …, 0.95). q10/q50/q90 are
  the served quantiles.
- **CRPS** is approximated from the quantiles as 2 × mean pinball loss over the 19 levels.
  Also reported: pinball loss at the served quantiles, 80% coverage (q10–q90), mean 80%
  width, MAPE on q50, bias.
- **Scored rows:** `has_gap` false, target > 0, and the model's required inputs present.
  Cross-region averages are reported with and without LDWP (§4).
- **Outputs:** `workspace.gold.backtest_predictions` (model, fold, region, hour, 19
  quantiles, actual) and `workspace.gold.backtest_metrics` (per model × fold × region). Each
  model replaces only its own rows, so models can be rerun independently.

**Leakage warning:** backtests that use actual weather are optimistic, because at serving
time only forecast weather is available. Either train and evaluate against archived
weather forecasts, or state this limitation explicitly in the README. Do not quietly
report the optimistic number.

**Weather leakage, measured (2026-10-17).** GAM trained on actual weather, scored on the
21 folds with three different temperatures at prediction time (excl. LDWP, 137,800 hours):

| Prediction-time weather | CRPS | vs seasonal-naive | MAPE | 80% coverage |
| --- | --- | --- | --- | --- |
| Actual (ERA5) | 1,224.8 | −52.3% | 4.53% | 76.0% |
| Open-Meteo `historical_forecast` | 1,223.7 | −52.4% | 4.59% | 76.3% |
| **Forecast issued a day earlier (`previous_day1`)** | **1,274.8** | **−50.4%** | **4.74%** | 76.3% |
| Seasonal-naive (reference) | 2,570.3 | — | 8.55% | 79.9% |

- `historical_forecast` scores like actual weather, so it **cannot measure leakage**: it is
  stitched from the start of each forecast run, far shorter lead than our 11–34h. Do not
  report it as the honest number.
- `previous_day1` (Open-Meteo Previous Runs API, ~24–47h lead, MAE 1.52 °C against ERA5,
  p99 miss 5.9 °C) is the honest input. **Report −50.4% as the headline**, and the gap to
  actual weather (+4.1% CRPS, +0.21 pts MAPE) as the measured cost of weather-forecast error.
- Its lead time is slightly longer than ours, so if anything it overstates the cost.
- GAM coverage (76%) is still below 80% in every variant, worst in June folds. Separate
  problem, being fixed next.

**GAM intervals: 365-day calibration + refit (decided 2026-10-17).** The base construction
(z learned on the last 90 days of training, held-out model used to forecast) covered 76%.
`scripts/eda_gam_coverage.py` compared six variants with honest weather (excl. LDWP):

| Variant | Coverage | Mean \|fold − 80\| | Worst fold | CRPS | Width |
| --- | --- | --- | --- | --- | --- |
| 90d calibration, z per hour (old default) | 76.3% | 4.40 | 65.5% | 1,274.8 | 5,310 |
| z pooled across hours | 80.2% | 3.00 | 74.0% | 1,296.1 | 5,683 |
| 365d calibration | 82.1% | 3.03 | 75.1% | 1,299.0 | 6,048 |
| 365d + pooled | 82.1% | 2.85 | 76.2% | 1,316.4 | 6,031 |
| refit only | 74.9% | 5.24 | 64.5% | 1,273.6 | 5,158 |
| **365d calibration + refit (new default)** | **80.0%** | **2.45** | 73.7% | **1,256.7** | 5,551 |

90 days captured one season's error shape (June folds learned from calm spring: 65–69%).
365 days alone over-covered because the held-out model was a year stale. Learning the shape
on a full year and then refitting for the forecast hits 80.0%, every region at 79.6–81.3%,
and also gives the best CRPS (fresher model, MAPE 4.66%). June folds remain the weakest
(73.7%, 76.2%): early-summer heat outruns the 7-day volatility scale. Say so in the README.

**Headline with the new default:** GAM with day-ahead forecast weather, CRPS 1,256.7 vs
2,570.3 for seasonal-naive (**51.1% better**), MAPE 4.66% vs 8.55%, coverage 80.0%.

### 7a. Measured findings on the seasonal-naive benchmark (2026-10-12)

From `databricks/eda_naive_error.py`, `eda_naive_extremes.py` and `eda_naive_dst.py`, CISO
unless stated. These are measurements, not estimates, and they shape the §7 regime
breakouts.

**Every heat wave costs seasonal-naive twice, a week apart, with opposite signs.** Because
the forecast is last week's value, a heat wave is first under-forecast on arrival and then
over-forecast seven days later when the heat has gone. September 2026 shows the pair
cleanly: +10,735 MW bias on 09-09 (40.5 °C, 49,959 MW peak), then −11,109 MW on 09-16
(28.1 °C). The same pattern repeats in September 2024. This is structural, not incidental:
since `e_t = y_t − y_{t−168}`, the term `y_t` enters `e_t` positively and `e_{t+168}`
negatively, so naive errors are negatively autocorrelated at lag 168 by construction. Two
consequences: the heat-wave regime in §7 is load-bearing, and the Diebold-Mariano tests must
use a HAC (Newey–West) variance estimator, because hourly loss differentials are strongly
dependent and treating them as independent would badly overstate significance.

**September is the hardest month, but quote a range, not a point.** 11.99% MAPE against a
4–8% norm. It is hard three ways at once: systematically (negative bias in all four years,
from the cooling-season decline), broadly (highest *median* hourly error of any month, 2,350
MW, so not merely tail-driven), and episodically (7.85% in 2025 versus 15.87% in 2026). With
only four Septembers, the maximum of twelve monthly estimates is biased upward by selection,
so any single September figure is unreliable.

**DST misalignment is real at the hour level but does not raise aggregate error.**
`lag_168h` steps back 168 hours of UTC, which equals one week of local clock time only while
the offset holds. For the seven days after a transition the benchmark compares against a
different local hour, and the signed effect is exactly as the mechanism predicts: after
spring forward it reaches an hour earlier and under-forecasts (+745 to +1,151 MW at hours
7–9); after fall back it reaches an hour later and over-forecasts (−1,081 to −1,597 MW at the
same hours). The damage concentrates in the **morning** ramp, not the evening — DST
transitions fall in mild months where there is no AC-driven evening peak, so the steepest
hourly change is 7–9am.

But aggregate error does *not* rise: 5.80% in spring-transition weeks and 4.13% in
fall-transition weeks, against 6.76% for aligned weeks. That comparison is confounded —
misaligned rows only ever occur in mid-March and early November. Against the correct
baseline (March 5.86%, November 4.31%) the effect is approximately zero. Eight of ten
regions show `penalty_x` below 1.0. **Report the null, with the mechanism and the
confounder.** Do not claim DST degrades the benchmark.

**Error does not track demand level.** CISO demand peaks at hour 20 (29,400 MW) but error
peaks at hour 17 (2,857 MW MAE, 10.39% MAPE, versus 7.72% at hour 20). Hour 17 is harder
despite being smaller, because behind-the-meter solar is collapsing and is highly sensitive
to cloud cover, while by hour 20 solar variability is zero. Hard hours run ~2.5× the easy
hours (5–7am, ~4.2%). Two implications: naive prediction intervals must be stratified by
local hour, and the §14a solar-radiation item is now evidence-backed rather than
speculative.

**Seasonal-naive intervals: volatility-scaled, 7-day window (decided 2026-10-15).** From
`eda_naive_intervals.py` → `eda_naive_drift.py` → `eda_naive_seasonal.py` →
`eda_naive_volscaled.py`, one train/test split (train 2023–24, test 2025-01 → 2026-09):

    q_τ(t) = lag_168h(t) + σ(t) · z_τ(region, hour_local)

σ(t) is the mean |naive error| over the 7 days ending 48h before t (known at issue time).
z_τ are quantiles of the standardised error e/σ, learned on training data only.

| Method | Mean \|coverage − 80\| over 21 months | Worst − best month |
| --- | --- | --- |
| Empirical offsets per (region, hour) | 8.85 pts | 36.7 pts |
| Best calendar stratification (season / month) | 7.29 | 24.8 |
| Volatility-scaled, 7-day σ | **2.35** | **10.9** |
| Volatility-scaled, 28-day σ | 4.16 | 18.9 |
| Volatility-scaled, 56-day σ | 6.32 | 25.3 |

Why it works: raw error spread grew 17% from train to test (demand growth), but the
standardised z spread changed by 0.6% (3.548 → 3.571). Scale moves with recent conditions;
shape is stable. The two Augusts that calendar methods could not fix (53.2% and 77.5% under
hour offsets) both land near 80% (78.6%, 81.2%), and every region lands at 78.7–81.9%.

Caveats to state in the README: (1) better coverage is partly bought with width (e.g. PJM's
mean 80% width rises from 25.4k to 31.7k MW at 28 days); CRPS in the 21-fold backtest is the
arbiter of that trade. (2) The window length was picked by comparing three options on the
same test months, a mild selection effect. The gap is large, and 7 days is now fixed in
advance of the backtest. The same scale × shape idea (normalised conformal prediction) is
the plan for the GAM's intervals too.

**Verified as a side effect:** gold's `lag_168h` equals the demand exactly 168 hours earlier
in every row checked (`lag_mismatches = 0`), so the window function in `03_gold_features.py`
is stepping the intended distance.

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
│   ├── eda_naive_error.py     # EDA, named outside the numbered sequence
│   ├── eda_naive_extremes.py  # mean vs median: broad difficulty or tail events
│   ├── eda_naive_dst.py       # DST misalignment, measured not hardcoded
│   └── databricks.yml         # Asset Bundle - jobs as code
├── models/                    # pure pandas/numpy, no Spark import, so the same code runs
│   ├── __init__.py            #   under applyInPandas and in the local harness
│   ├── scoring.py             # TAUS, CRPS/pinball/coverage - one definition for all models
│   ├── intervals.py           # volatility-scaled intervals (section 7a), scale x shape
│   ├── naive.py               # seasonal-naive, the local twin of 06_backtest.py
│   └── gam.py                 # pygam LinearGAM spec, fit, and calibrated quantiles
├── service/
│   ├── api.py                 # FastAPI app
│   ├── model_loader.py        # fetch registered artifact
│   ├── features.py            # serving-time feature construction
│   └── Dockerfile             # multi-stage
├── tests/
├── scripts/
│   ├── check_local_env.py     # day-0 environment doctor
│   ├── smoke.py               # serving-contract assertions, stdlib only
│   ├── export_gold.py         # snapshot the modeling tables to data/*.parquet
│   ├── backtest_local.py      # the 21-fold backtest in pandas, no Databricks compute
│   └── gam_prototype.py       # one region, one fold, with term plots
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

**Missing inputs at serving time (decided 2026-10-10, to be built on day 7).** EIA sometimes
publishes no demand for a whole local day, and several of these days fall on DST
transitions (e.g. PJM on 2024-03-10, confirmed blank at the source). A regression-style model
can't predict with a missing input, so when a lag feature is null at issue time:

1. Use the **same hour on the most recent earlier day that has data** (72h back, then 96h,
   and so on). This keeps the hour-of-day shape, which a rolling mean throws away (a 168h
   mean runs 15–20% low at the 5 PM peak).
2. Use the rolling mean only as a last resort, if several days in a row are missing.
3. Log every fallback in the prediction log so its effect can be scored.

Validate on day 7 with a backtest: blank out lags on some days, fill them each way, and
compare error. Next DST change is 2026-11-01, inside the monitoring window, so it's a live
test.

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
