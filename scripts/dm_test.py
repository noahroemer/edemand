"""Diebold-Mariano test: is the GAM's CRPS improvement over seasonal-naive real?

For each hour t, the loss differential is d_t = CRPS_naive(t) - CRPS_gam(t). The test asks
whether mean(d) is distinguishable from zero. Hourly losses are strongly autocorrelated
(weather persists for days, and naive errors are autocorrelated at lag 168 by construction,
CLAUDE.md section 7a), so the variance of mean(d) uses a Newey-West (HAC) estimator with a
one-week lag window. Treating hours as independent would overstate significance badly.

Reads data/local_predictions.parquet (scripts/backtest_local.py output).

    python scripts/dm_test.py [MODEL]          # default gam_fcst
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.scoring import TAUS  # noqa: E402

DATA = REPO / "data"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gam_fcst"
BENCHMARK = "seasonal_naive"
HAC_LAG = 168  # one week, matching the seasonal-naive error structure


def hourly_crps(preds: pd.DataFrame) -> pd.Series:
    q = np.stack(preds.quantiles.to_numpy())
    diff = preds.actual.to_numpy()[:, None] - q
    pinball = np.where(diff >= 0, TAUS * diff, (TAUS - 1) * diff)
    return pd.Series(2 * pinball.mean(axis=1), index=preds.index)


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of d with Bartlett weights."""
    d = d - d.mean()
    n = len(d)
    var = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k] / n)
    return var


def dm(d: np.ndarray) -> tuple[float, float, float]:
    """(mean d, DM statistic, two-sided p-value)."""
    se = np.sqrt(newey_west_var(d, HAC_LAG) / len(d))
    stat = d.mean() / se
    return d.mean(), stat, 2 * stats.norm.sf(abs(stat))


preds = pd.read_parquet(DATA / "local_predictions.parquet")
preds = preds[preds.model.isin([MODEL, BENCHMARK])].copy()
preds["crps"] = hourly_crps(preds)
wide = preds.pivot_table(index=["region", "period_utc"], columns="model", values="crps").dropna()
wide["d"] = wide[BENCHMARK] - wide[MODEL]

rows = []
for region, g in wide.groupby(level="region"):
    mean_d, stat, p = dm(g.sort_index().d.to_numpy())
    rows.append(
        {
            "region": region,
            "hours": len(g),
            "crps_naive": g[BENCHMARK].mean(),
            f"crps_{MODEL}": g[MODEL].mean(),
            "mean_d": mean_d,
            "dm_stat": stat,
            "p_value": p,
        }
    )

# Pooled across regions (excl. LDWP): average d over regions each hour, then test the series.
pooled = wide[wide.index.get_level_values("region") != "LDWP"].groupby(level="period_utc").d.mean()
mean_d, stat, p = dm(pooled.sort_index().to_numpy())
rows.append(
    {
        "region": "ALL excl. LDWP",
        "hours": len(pooled),
        "crps_naive": np.nan,
        f"crps_{MODEL}": np.nan,
        "mean_d": mean_d,
        "dm_stat": stat,
        "p_value": p,
    }
)

table = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(f"=== Diebold-Mariano: {BENCHMARK} vs {MODEL}, CRPS loss, Newey-West lag {HAC_LAG}h ===")
print("mean_d > 0 means the model beats the benchmark. |DM| > 1.96 is significant at 5%.")
print(
    table.to_string(
        index=False,
        formatters={
            "crps_naive": "{:,.1f}".format,
            f"crps_{MODEL}": "{:,.1f}".format,
            "mean_d": "{:,.1f}".format,
            "dm_stat": "{:.2f}".format,
            "p_value": "{:.1e}".format,
        },
    )
)
