"""Walk-forward backtest run locally over data/*.parquet -- no Databricks compute.

Same folds, same quantile levels, same metric definitions as databricks/06_backtest.py
(CLAUDE.md section 7), but reading the exported snapshot and looping in pandas. The models
themselves live in models/, so the code validated here is the code the Spark fleet job
will run under applyInPandas -- there is no prototype-to-production rewrite.

Why local: 210 GAM fits (10 regions x 21 folds) is real Free Edition compute, and the
quota has already run out twice. A fit takes well under a second here, so model choices
get iterated for free; the official run still happens on Databricks.

The seasonal-naive benchmark is recomputed rather than read from the export, for two
reasons: the export only carries q10/q50/q90, so CRPS could not be compared without it,
and running both implementations on the same rows cross-checks the Spark one (printed).

    python scripts/backtest_local.py                        # everything
    python scripts/backtest_local.py --regions CISO --folds 1 8
    python scripts/backtest_local.py --models gam
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # import models/ without installing the package

from models import Fold, gam, naive  # noqa: E402
from models.scoring import I10, I50, I90, metrics_by  # noqa: E402

DATA = REPO / "data"
OUT = DATA / "local_predictions.parquet"
MODELS = {
    "seasonal_naive": naive.fit_predict,
    "gam": gam.fit_predict,  # predicts with actual weather: optimistic
    "gam_hf": partial(gam.fit_predict, weather="temp_c_hf"),  # historical_forecast: ~actual
    "gam_fcst": partial(gam.fit_predict, weather="temp_c_pd1"),  # forecast a day earlier: honest
}
FORECAST_COLUMNS = {"historical_forecast": "temp_c_hf", "previous_day1": "temp_c_pd1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="one line per model, not per fit")
    return p.parse_args()


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


def run(args: argparse.Namespace) -> pd.DataFrame:
    # Forecast weather arrives long (one row per region, source, hour); widen it to one
    # temperature column per source, so each model variant just names the column it uses.
    forecasts = (
        pd.read_parquet(DATA / "forecast_weather.parquet")
        .pivot_table(index=["region", "period_utc"], columns="source", values="temp_c")
        .rename(columns=FORECAST_COLUMNS)
        .reset_index()
    )
    gold = pd.read_parquet(DATA / "gold_features.parquet").merge(
        forecasts, on=["region", "period_utc"], how="left"
    )
    folds = [
        Fold.from_row(r) for r in pd.read_parquet(DATA / "backtest_folds.parquet").itertuples()
    ]
    if args.folds:
        folds = [f for f in folds if f.fold in args.folds]
    regions = args.regions or sorted(gold.region.unique())

    frames = []
    for name in args.models:
        fit_predict = MODELS[name]
        started = time.perf_counter()
        for region in regions:
            history = gold[gold.region == region]
            for fold in folds:
                t0 = time.perf_counter()
                out = fit_predict(history, fold)
                out.insert(0, "region", region)
                out.insert(0, "fold", fold.fold)
                out.insert(0, "model", name)
                frames.append(out)
                if not args.quiet:
                    print(
                        f"{name:15s} {region:5s} fold {fold.fold:2d} "
                        f"{fold.test_start:%Y-%m}  {len(out):4d} hours  "
                        f"{time.perf_counter() - t0:5.2f}s",
                        flush=True,
                    )
        print(f"{name}: {time.perf_counter() - started:.1f}s total")

    preds = pd.concat(frames, ignore_index=True)
    quantiles = np.stack(preds.quantiles.to_numpy())
    preds["q10"], preds["q50"], preds["q90"] = (quantiles[:, i] for i in (I10, I50, I90))
    return preds


def cross_check(preds: pd.DataFrame) -> None:
    """Compare the local seasonal-naive against the Spark run exported to data/."""
    spark_run = DATA / "naive_predictions.parquet"
    local = preds[preds.model == "seasonal_naive"]
    if local.empty or not spark_run.exists():
        return
    keys = ["fold", "region", "period_utc"]
    both = local.merge(pd.read_parquet(spark_run), on=keys, suffixes=("", "_spark"))
    print(f"\n=== Local vs Spark seasonal-naive: {len(both):,} of {len(local):,} rows matched ===")
    for col in ["q10", "q50", "q90", "actual"]:
        gap = (both[col] - both[f"{col}_spark"]).abs()
        relative = 100 * gap.max() / both[col].abs().mean()
        print(f"  {col}: max |diff| {gap.max():8.2f} MW ({relative:.3f}% of mean level)")
    print("  q50 should match exactly; q10/q90 differ only by percentile_approx's tolerance.")


def report(preds: pd.DataFrame) -> None:
    models = sorted(preds.model.unique())
    if len(models) > 1:
        # Models drop different rows (each needs its own inputs present), so compare on the
        # hours every model scored. Diebold-Mariano later needs the same matched set.
        scored = preds.groupby(["region", "fold", "period_utc"], sort=False).model.transform("size")
        common = preds[scored == len(models)]
        print(
            f"\nCommon rows: {len(common) // len(models):,} hours per model, of {len(preds) // len(models):,}"
        )
    else:
        common = preds

    show(
        metrics_by(common, ["model", "region"]).sort_values(["region", "model"]),
        "Pooled over all folds, by region",
    )

    overall = pd.concat(
        [
            metrics_by(common.assign(subset="all 10"), ["subset", "model"]),
            metrics_by(
                common[common.region != "LDWP"].assign(subset="excl. LDWP"), ["subset", "model"]
            ),
        ]
    )
    show(overall, "All regions, with and without LDWP (CLAUDE.md section 4)")

    no_ldwp = common[common.region != "LDWP"]
    by_fold = metrics_by(no_ldwp, ["fold", "model"]).pivot(
        index="fold", columns="model", values=["crps", "coverage80"]
    )
    by_fold.columns = [f"{metric}_{model}" for metric, model in by_fold.columns]
    show(by_fold.reset_index(), "By fold, excluding LDWP (coverage target 80)")

    if "seasonal_naive" in models:
        wide = metrics_by(common, ["region", "model"]).pivot(index="region", columns="model")
        for model in [m for m in models if m != "seasonal_naive"]:
            gain = pd.DataFrame(
                {
                    "crps_naive": wide[("crps", "seasonal_naive")],
                    f"crps_{model}": wide[("crps", model)],
                    "crps_better_pct": 100
                    * (1 - wide[("crps", model)] / wide[("crps", "seasonal_naive")]),
                    "mape_naive": wide[("mape", "seasonal_naive")],
                    f"mape_{model}": wide[("mape", model)],
                    f"coverage_{model}": wide[("coverage80", model)],
                }
            ).round(2)
            show(gain.reset_index(), f"{model} vs seasonal-naive (CRPS lower is better)")


def main() -> None:
    args = parse_args()
    preds = run(args)
    preds.to_parquet(OUT, index=False)
    print(f"\nwrote {len(preds):,} rows to {OUT}")
    cross_check(preds)
    report(preds)


if __name__ == "__main__":
    main()
