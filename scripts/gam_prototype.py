"""GAM prototype: one region, one fold, entirely local (reads data/ from export_gold.py).

The point is to see the model work and compare it with seasonal-naive on the same month
before building the 210-fit Databricks version (10 regions x 21 folds).

Model: demand as a sum of smooth curves (pygam LinearGAM)

    s(temp_c)               temperature response, learned (expect a U shape)
    s(hour_of_week, cyclic) 0..167, so Saturday's daily shape can differ from Tuesday's
    s(day_of_year, cyclic)  seasons
    f(is_major_holiday)     a level shift on major holidays
    l(lag_168h), l(lag_48h) recent demand level

Intervals: the same volatility-scaled method as seasonal-naive (CLAUDE.md section 7a),
applied to GAM residuals. The training window is split so the interval shape is learned
from honest out-of-sample residuals, not in-sample ones (which are too small):

    fit window        train_start .. train_end - 90 days   -> fit the GAM
    calibration       last 90 days of training             -> z quantiles of residual / sigma
    test              the fold's test month                -> q_tau = yhat + sigma * z_tau

    python scripts/gam_prototype.py [REGION] [FOLD]
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s

DATA = Path(__file__).resolve().parent.parent / "data"
REGION = sys.argv[1] if len(sys.argv) > 1 else "CISO"
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WEATHER = sys.argv[3] if len(sys.argv) > 3 else "archive"
"""Weather used at PREDICTION time. Training always uses archive (actual) weather, as it
would in production. "archive" predicts with actual weather too: optimistic, because at
issue time only a forecast exists. "forecast" predicts with the forecast as issued: the
honest number (CLAUDE.md section 7 leakage warning)."""

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
I10, I50, I90 = 1, 9, 17
CALIBRATION_DAYS = 90
EMBARGO_H, SIGMA_DAYS = 48, 7
FEATURES = ["temp_c", "hour_of_week", "day_of_year", "is_major_holiday", "lag_168h", "lag_48h"]

# --- Data ---------------------------------------------------------------------------------

gold = pd.read_parquet(DATA / "gold_features.parquet")
row = pd.read_parquet(DATA / "backtest_folds.parquet").set_index("fold").loc[FOLD]
fold = SimpleNamespace(**{k: pd.Timestamp(v) for k, v in row.items()})

df = gold[gold.region == REGION].sort_values("period_utc").copy()
df["hour_of_week"] = df.dow * 24 + df.hour_local
df["day_of_year"] = pd.to_datetime(df.local_date).dt.dayofyear
df["is_major_holiday"] = df.is_major_holiday.astype(int)
usable = ~df.has_gap & (df.demand_mw > 0) & df[FEATURES].notna().all(axis=1)
df = df[usable]

calib_start = fold.train_end - pd.Timedelta(days=CALIBRATION_DAYS)
fit = df[(df.period_utc >= fold.train_start) & (df.period_utc < calib_start)]
calib = df[(df.period_utc >= calib_start) & (df.period_utc < fold.train_end)]
test = df[(df.period_utc >= fold.test_start) & (df.period_utc < fold.test_end)]
print(f"{REGION} fold {FOLD}: fit {len(fit):,} rows, calibrate {len(calib):,}, test {len(test):,}")
print(f"test month {fold.test_start:%Y-%m}, prediction weather: {WEATHER}")

# --- Fit ----------------------------------------------------------------------------------

gam = LinearGAM(
    s(0, n_splines=15)  # temp_c
    + s(1, n_splines=60, basis="cp", edge_knots=[0, 168])  # hour_of_week, wraps 167 -> 0
    + s(2, n_splines=12, basis="cp", edge_knots=[1, 367])  # day_of_year, wraps Dec -> Jan
    + f(3)  # is_major_holiday
    + l(4)  # lag_168h
    + l(5)  # lag_48h
)
gam.fit(fit[FEATURES].to_numpy(), fit.demand_mw.to_numpy())

# --- Intervals: volatility-scaled residuals -----------------------------------------------
# Predict calibration + test as one continuous series, so sigma (recent |residual|, ending
# 48h back) is available from the first test hour. Residuals 48h+ old are known at issue time.

# Includes the 48h embargo between training and test: those hours aren't trained on, but
# their residuals are known by the time they feed sigma (48h later).
series = df[(df.period_utc >= calib_start) & (df.period_utc < fold.test_end)].copy()
if WEATHER == "forecast":
    fc = pd.read_parquet(DATA / "forecast_weather.parquet")
    fc = fc[fc.region == REGION][["period_utc", "temp_c_fc"]]
    series = series.merge(fc, on="period_utc", how="left")
    series["temp_c"] = series.temp_c_fc
    series = series.dropna(subset=["temp_c"])
series["yhat"] = gam.predict(series[FEATURES].to_numpy())
series["abs_resid"] = (series.demand_mw - series.yhat).abs()
series = series.set_index("period_utc")
window = f"{SIGMA_DAYS * 24}h"
series["sigma"] = (
    series.abs_resid.shift(freq=f"{EMBARGO_H}h").rolling(window).mean().reindex(series.index)
)
series = series.reset_index()

cal = series[series.period_utc < fold.train_end].dropna(subset=["sigma"])
z = (cal.demand_mw - cal.yhat) / cal.sigma
z_q = cal.assign(z=z).groupby("hour_local").z.quantile(TAUS).unstack()  # hour x tau

out = series[series.period_utc >= fold.test_start].dropna(subset=["sigma"]).copy()
q = (
    out.yhat.to_numpy()[:, None]
    + out.sigma.to_numpy()[:, None] * z_q.loc[out.hour_local].to_numpy()
)

# --- Score, same definitions as 06_backtest.py --------------------------------------------


def score(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    diff = y[:, None] - q
    pinball = np.where(diff >= 0, TAUS * diff, (TAUS - 1) * diff)
    return {
        "crps": 2 * pinball.mean(),
        "mape": 100 * np.mean(np.abs(y - q[:, I50]) / y),
        "coverage80": 100 * np.mean((y >= q[:, I10]) & (y <= q[:, I90])),
        "width80": np.mean(q[:, I90] - q[:, I10]),
    }


y = out.demand_mw.to_numpy()
gam_scores = score(y, q)

naive = pd.read_parquet(DATA / "naive_predictions.parquet")
naive = naive[(naive.region == REGION) & (naive.fold == FOLD)]
both = out[["period_utc"]].merge(naive, on="period_utc")
print(f"\nscored hours: GAM {len(out):,}, naive {len(naive):,}, both {len(both):,}")
naive_mape = 100 * np.mean(np.abs(both.actual - both.q50) / both.actual)
naive_cov = 100 * np.mean((both.actual >= both.q10) & (both.actual <= both.q90))

print(f"\n{'':14s}{'GAM':>10s}{'naive':>10s}")
print(f"{'MAPE %':14s}{gam_scores['mape']:>10.2f}{naive_mape:>10.2f}")
print(f"{'coverage80 %':14s}{gam_scores['coverage80']:>10.1f}{naive_cov:>10.1f}")
print(f"{'CRPS (MW)':14s}{gam_scores['crps']:>10.0f}{'see below':>10s}")
print(f"{'width80 (MW)':14s}{gam_scores['width80']:>10.0f}")
print("(naive CRPS for this fold is in gold.backtest_metrics; 19 quantiles weren't exported)")

# --- Plots --------------------------------------------------------------------------------

plots = DATA / "plots"
plots.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, term, title in zip(
    axes, [0, 1, 2], ["temperature (C)", "hour of week (0 = Mon 00:00)", "day of year"], strict=True
):
    grid = gam.generate_X_grid(term=term)
    pdep, ci = gam.partial_dependence(term=term, X=grid, width=0.95)
    ax.plot(grid[:, term], pdep)
    ax.fill_between(grid[:, term], ci[:, 0], ci[:, 1], alpha=0.25)
    ax.set_title(f"{REGION}: effect of {title}")
    ax.set_ylabel("MW (relative)")
fig.tight_layout()
fig.savefig(plots / f"gam_terms_{REGION}_fold{FOLD}.png", dpi=110)

week = out.period_utc < out.period_utc.min() + pd.Timedelta(days=7)
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(out.period_utc[week], q[week, I10], q[week, I90], alpha=0.3, label="80% interval")
ax.plot(out.period_utc[week], q[week, I50], label="GAM median")
ax.plot(out.period_utc[week], y[week], color="black", lw=1, label="actual")
ax.set_title(f"{REGION}: first test week, fold {FOLD}")
ax.legend()
fig.tight_layout()
fig.savefig(plots / f"gam_week_{REGION}_fold{FOLD}.png", dpi=110)
print(f"\nplots saved to {plots}")
