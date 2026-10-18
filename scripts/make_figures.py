"""README figures from the local backtest (data/local_predictions.parquet).

    docs/figures/forecast_week.png         GAM forecast vs actual for one week, intervals shaded
    docs/figures/calibration.png           nominal vs empirical coverage at all 19 quantiles
    docs/figures/temperature_response.png  the GAM's learned temperature curve per region

(error_by_regime.png comes from scripts/regime_analysis.py.)

    python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models import Fold, gam  # noqa: E402
from models.scoring import TAUS  # noqa: E402

DATA = REPO / "data"
FIGURES = REPO / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
GAM, NAIVE = "gam_fcst", "seasonal_naive"
I25, I75 = int(np.where(TAUS == 0.25)[0][0]), int(np.where(TAUS == 0.75)[0][0])

preds = pd.read_parquet(DATA / "local_predictions.parquet")
gold = pd.read_parquet(DATA / "gold_features.parquet")
if "parent_region" in gold:
    gold = gold[gold.parent_region.isna()]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# --- 1. A week of forecasts ---------------------------------------------------------------
# CISO's September 2026 heat wave: the hardest stretch for seasonal-naive (CLAUDE.md 7a).
REGION, START = "CISO", pd.Timestamp("2026-09-05")
week = preds[
    (preds.region == REGION)
    & (preds.period_utc >= START)
    & (preds.period_utc < START + pd.Timedelta(days=7))
]
g = week[week.model == GAM].sort_values("period_utc")
n = week[week.model == NAIVE].sort_values("period_utc")
q = np.stack(g.quantiles.to_numpy())
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(g.period_utc, g.q10, g.q90, alpha=0.2, color="C0", label="GAM 80% interval")
ax.fill_between(
    g.period_utc, q[:, I25], q[:, I75], alpha=0.35, color="C0", label="GAM 50% interval"
)
ax.plot(g.period_utc, g.q50, color="C0", lw=1.6, label="GAM median")
ax.plot(n.period_utc, n.q50, color="C1", lw=1.2, ls="--", label="Seasonal-naive (last week)")
ax.plot(g.period_utc, g.actual, color="black", lw=1.2, label="Actual")
ax.set_title(
    f"{REGION}, week of {START:%b %d %Y} (heat wave), forecasts made with day-ahead weather"
)
ax.set_ylabel("Demand (MW)")
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "forecast_week.png", dpi=130)

# --- 2. Calibration across all quantile levels -------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect calibration")
for model, label in ((NAIVE, "Seasonal-naive"), (GAM, "GAM (day-ahead weather)")):
    m = preds[(preds.model == model) & (preds.region != "LDWP")]
    q = np.stack(m.quantiles.to_numpy())
    empirical = (m.actual.to_numpy()[:, None] <= q).mean(axis=0)
    ax.plot(TAUS, empirical, marker="o", ms=4, label=label)
ax.set_xlabel("Nominal quantile level")
ax.set_ylabel("Share of actuals at or below the forecast quantile")
ax.set_title("Calibration, 9 regions, 21 folds")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "calibration.png", dpi=130)

# --- 3. Learned temperature response ------------------------------------------------------
# Fit on the last fold's full training window; the temperature term's partial dependence is
# the U-shaped response. Scaled to % of mean demand so regions of any size share an axis.
last = Fold.from_row(pd.read_parquet(DATA / "backtest_folds.parquet").iloc[-1])
fig, ax = plt.subplots(figsize=(8, 5))
for region in ["ERCO", "CISO", "PJM", "ISNE", "FPL", "BPAT"]:
    history = gold[gold.region == region]
    model, rows = gam.fit(history, last, calibration_days=0)
    lo, hi = rows.temp_c.quantile([0.005, 0.995])
    grid = model.generate_X_grid(term=0, n=200)
    grid = grid[(grid[:, 0] >= lo) & (grid[:, 0] <= hi)]
    effect, ci = model.partial_dependence(term=0, X=grid, width=0.95)
    scale = 100 / rows.demand_mw.mean()
    shift = effect.min()
    line = ax.plot(grid[:, 0], (effect - shift) * scale, label=region)[0]
    ax.fill_between(
        grid[:, 0],
        (ci[:, 0] - shift) * scale,
        (ci[:, 1] - shift) * scale,
        alpha=0.15,
        color=line.get_color(),
    )
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Effect on demand (% of mean, relative to minimum)")
ax.set_title("Learned temperature response (GAM, 95% bands): demand rises in heat and cold")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "temperature_response.png", dpi=130)

print(f"saved figures to {FIGURES}")
