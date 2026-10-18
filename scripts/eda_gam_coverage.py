"""Why do GAM intervals cover 76% instead of 80%? Compare interval variants, measured.

The point forecast is fixed (GAM, honest day-ahead weather); only the interval
construction changes. Three suspects, each with a switch in models/gam.fit_predict:

    A  calibration window too short / one season  -> calibration_days = 365
    B  too few points per tail quantile            -> z_by_hour = False (pool hours)
    C  stale model: fit ends 90 days before train   -> refit = True

Caveat on C: after refitting, sigma for the first test days averages residuals from the
last training week, which are now in-sample (too small). Expect C to under-cover early in
each month for that reason, separate from whether staleness matters.

Local only, reads data/*.parquet.

    python scripts/eda_gam_coverage.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models import Fold, gam  # noqa: E402
from models.scoring import metrics_by  # noqa: E402

DATA = REPO / "data"
WEATHER = "temp_c_pd1"

VARIANTS = {
    "base (90d, by hour)": {},
    "B pooled hours": {"z_by_hour": False},
    "A 365d calib": {"calibration_days": 365},
    "A+B 365d pooled": {"calibration_days": 365, "z_by_hour": False},
    "C refit": {"refit": True},
    "A+C 365d refit": {"calibration_days": 365, "refit": True},
}


def load() -> tuple[pd.DataFrame, list[Fold]]:
    forecasts = (
        pd.read_parquet(DATA / "forecast_weather.parquet")
        .query("source == 'previous_day1'")
        .rename(columns={"temp_c": WEATHER})[["region", "period_utc", WEATHER]]
    )
    gold = pd.read_parquet(DATA / "gold_features.parquet").merge(
        forecasts, on=["region", "period_utc"], how="left"
    )
    if "parent_region" in gold:
        gold = gold[gold.parent_region.isna()]
    folds = [
        Fold.from_row(r) for r in pd.read_parquet(DATA / "backtest_folds.parquet").itertuples()
    ]
    return gold, folds


def run_region(region: str) -> pd.DataFrame:
    gold, folds = load()
    history = gold[gold.region == region]
    frames = []
    for name, options in VARIANTS.items():
        for fold in folds:
            out = gam.fit_predict(history, fold, weather=WEATHER, **options)
            out.insert(0, "fold", fold.fold)
            out.insert(0, "region", region)
            out.insert(0, "variant", name)
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    gold, _ = load()
    regions = sorted(gold.region.unique())
    with ProcessPoolExecutor(max_workers=5) as pool:
        preds = pd.concat(pool.map(run_region, regions), ignore_index=True)
    preds.to_parquet(DATA / "gam_coverage_variants.parquet", index=False)

    clean = preds[preds.region != "LDWP"]
    overall = metrics_by(clean, ["variant"]).set_index("variant")
    by_fold = metrics_by(clean, ["variant", "fold"])
    gap = by_fold.assign(gap=(by_fold.coverage80 - 80).abs()).groupby("variant")
    by_region = metrics_by(clean, ["variant", "region"]).pivot(
        index="region", columns="variant", values="coverage80"
    )

    summary = pd.DataFrame(
        {
            "coverage80": overall.coverage80,
            "mean_abs_gap_by_fold": gap.gap.mean(),
            "worst_fold": gap.coverage80.min(),
            "crps": overall.crps,
            "width80": overall.width80,
            "mape": overall.mape,
        }
    ).loc[list(VARIANTS)]
    pd.set_option("display.width", 160)
    print("=== Excluding LDWP, all 21 folds (target coverage 80) ===")
    print(summary.round(2).to_string())
    print("\n=== Coverage by region ===")
    print(by_region[list(VARIANTS)].round(1).to_string())
    june = by_fold[by_fold.fold.isin([6, 18])].pivot(
        index="variant", columns="fold", values="coverage80"
    )
    print("\n=== June folds (6 = Jun 2025, 18 = Jun 2026), the worst under the base ===")
    print(june.loc[list(VARIANTS)].round(1).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
