"""GAM on one region and one fold, with the plots that show what the model learned.

The model itself now lives in models/gam.py -- this script only fits it, scores it against
seasonal-naive on the same hours, and draws. Keeping one definition matters: the Spark
fleet job, the local 21-fold harness and these plots must all be the same model.

Two plots:
  gam_terms_<REGION>_fold<N>.png   each smooth term on its own -- the temperature U, the
                                   weekly shape, the seasonal curve, with uncertainty
  gam_week_<REGION>_fold<N>.png    the first test week, 80% interval shaded

    python scripts/gam_prototype.py [REGION] [FOLD]
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display on this box; write PNGs straight to data/plots
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # import models/ without installing the package

from models import Fold, gam  # noqa: E402
from models.scoring import I10, I50, I90, metrics  # noqa: E402

DATA = REPO / "data"
REGION = sys.argv[1] if len(sys.argv) > 1 else "CISO"
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 1

TERMS = [(0, "temperature (C)"), (1, "hour of week (0 = Mon 00:00)"), (2, "day of year")]

gold = pd.read_parquet(DATA / "gold_features.parquet")
folds = pd.read_parquet(DATA / "backtest_folds.parquet").set_index("fold")
fold = Fold.from_row(folds.loc[FOLD].to_dict() | {"fold": FOLD})
history = gold[gold.region == REGION]

preds = gam.fit_predict(history, fold)
fitted, _ = gam.fit(history, fold)
quantiles = np.stack(preds.quantiles.to_numpy())
actual = preds.actual.to_numpy()

print(f"{REGION} fold {FOLD} ({fold.test_start:%Y-%m}): {len(preds):,} scored hours")
scores = metrics(actual, quantiles)

naive = pd.read_parquet(DATA / "naive_predictions.parquet")
naive = naive[(naive.region == REGION) & (naive.fold == FOLD)]
both = preds[["period_utc"]].merge(naive, on="period_utc")
print(f"seasonal-naive rows matched: {len(both):,}")

print(f"\n{'':14s}{'GAM':>10s}{'naive':>10s}")
print(
    f"{'MAPE %':14s}{scores['mape']:>10.2f}{100 * np.mean(np.abs(both.actual - both.q50) / both.actual):>10.2f}"
)
print(
    f"{'coverage80 %':14s}{scores['coverage80']:>10.1f}{100 * np.mean((both.actual >= both.q10) & (both.actual <= both.q90)):>10.1f}"
)
print(f"{'width80 (MW)':14s}{scores['width80']:>10.0f}{np.mean(both.q90 - both.q10):>10.0f}")
print(
    f"{'CRPS (MW)':14s}{scores['crps']:>10.0f}{'':>10s}  (naive CRPS needs all 19 quantiles: scripts/backtest_local.py)"
)

plots = DATA / "plots"
plots.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, len(TERMS), figsize=(15, 4))
for ax, (term, title) in zip(axes, TERMS, strict=True):
    grid = fitted.generate_X_grid(term=term)
    partial, ci = fitted.partial_dependence(term=term, X=grid, width=0.95)
    ax.plot(grid[:, term], partial)
    ax.fill_between(grid[:, term], ci[:, 0], ci[:, 1], alpha=0.25)
    ax.set_title(f"{REGION}: effect of {title}")
    ax.set_ylabel("MW (relative)")
fig.tight_layout()
fig.savefig(plots / f"gam_terms_{REGION}_fold{FOLD}.png", dpi=110)

week = preds.period_utc < pd.Timestamp(preds.period_utc.min()) + pd.Timedelta(days=7)
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(
    preds.period_utc[week],
    quantiles[week, I10],
    quantiles[week, I90],
    alpha=0.3,
    label="80% interval",
)
ax.plot(preds.period_utc[week], quantiles[week, I50], label="GAM median")
ax.plot(preds.period_utc[week], actual[week], color="black", lw=1, label="actual")
ax.set_title(f"{REGION}: first test week, fold {FOLD}")
ax.legend()
fig.tight_layout()
fig.savefig(plots / f"gam_week_{REGION}_fold{FOLD}.png", dpi=110)
print(f"\nplots saved to {plots}")
