"""Error by regime: where each model fails, and where the GAM earns its advantage.

Regimes are defined per region, so "heat wave" means unusually hot for that place:

    heat wave     local day whose max temperature is in the region's top 5% of days
    cold snap     local day whose min temperature is in the region's bottom 5% of days
    holiday       major holiday (calendars.major_holiday_dates)
    weekend       Saturday/Sunday, not in the regimes above
    weekday       everything else ("normal")

Thresholds use actual (ERA5) temperature over the full history, which is fine here: the
regime label is for slicing results, not a model input. Forecasts are scored as made, with
day-ahead weather. LDWP is excluded (CLAUDE.md section 4).

Reads data/local_predictions.parquet + data/gold_features.parquet. Writes
docs/figures/error_by_regime.png.

    python scripts/regime_analysis.py
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

from models.scoring import metrics_by  # noqa: E402

DATA = REPO / "data"
FIGURES = REPO / "docs" / "figures"
MODELS = {"seasonal_naive": "Seasonal-naive", "gam_fcst": "GAM (day-ahead weather)"}
ORDER = ["heat wave", "cold snap", "holiday", "weekend", "weekday"]

gold = pd.read_parquet(DATA / "gold_features.parquet")
if "parent_region" in gold:
    gold = gold[gold.parent_region.isna()]

# --- Label each local day -----------------------------------------------------------------
daily = gold.groupby(["region", "local_date"]).agg(
    tmax=("temp_c", "max"), tmin=("temp_c", "min"), holiday=("is_major_holiday", "max")
)
hot = daily.groupby("region").tmax.transform(lambda s: s >= s.quantile(0.95))
cold = daily.groupby("region").tmin.transform(lambda s: s <= s.quantile(0.05))
daily["regime"] = np.select([hot, cold, daily.holiday], ["heat wave", "cold snap", "holiday"], "")
labels = gold[["region", "period_utc", "local_date", "dow"]].merge(
    daily.regime.reset_index(), on=["region", "local_date"]
)
labels["regime"] = labels.regime.where(
    labels.regime != "", np.where(labels.dow >= 5, "weekend", "weekday")
)

# --- Score by regime ----------------------------------------------------------------------
preds = pd.read_parquet(DATA / "local_predictions.parquet")
preds = preds[preds.model.isin(MODELS) & (preds.region != "LDWP")]
preds = preds.merge(labels[["region", "period_utc", "regime"]], on=["region", "period_utc"])

table = metrics_by(preds, ["regime", "model"])
wide = table.pivot(index="regime", columns="model", values=["crps", "mape", "coverage80", "n"])
summary = pd.DataFrame(
    {
        "hours": wide[("n", "gam_fcst")].astype(int),
        "mape_naive": wide[("mape", "seasonal_naive")],
        "mape_gam": wide[("mape", "gam_fcst")],
        "crps_naive": wide[("crps", "seasonal_naive")],
        "crps_gam": wide[("crps", "gam_fcst")],
        "crps_better_pct": 100
        * (1 - wide[("crps", "gam_fcst")] / wide[("crps", "seasonal_naive")]),
        "coverage_naive": wide[("coverage80", "seasonal_naive")],
        "coverage_gam": wide[("coverage80", "gam_fcst")],
    }
).loc[ORDER]
pd.set_option("display.width", 160)
print("=== Error by regime, 9 regions (excl. LDWP), 21 folds ===")
print(summary.round(2).to_string())

by_region = metrics_by(preds[preds.model == "gam_fcst"], ["region", "regime"]).pivot(
    index="region", columns="regime", values="mape"
)[ORDER]
print("\n=== GAM MAPE % by region and regime ===")
print(by_region.round(2).to_string())

# --- Plot ---------------------------------------------------------------------------------
FIGURES.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(ORDER))
width = 0.38
for ax, metric, title in (
    (axes[0], "mape", "MAPE (%) by regime"),
    (axes[1], "coverage80", "80% interval coverage by regime"),
):
    for i, (model, label) in enumerate(MODELS.items()):
        values = [wide.loc[r, (metric, model)] for r in ORDER]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{r}\n({wide.loc[r, ('n', 'gam_fcst')]:,.0f} h)" for r in ORDER])
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].axhline(80, color="grey", lw=1, ls="--")
axes[1].set_ylim(50, 100)
axes[0].legend(frameon=False)
fig.suptitle("Where the models fail: 9 regions, Jan 2025 - Sep 2026, day-ahead weather")
fig.tight_layout()
out = FIGURES / "error_by_regime.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
