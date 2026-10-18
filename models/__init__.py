"""Model code shared by the local backtest harness and the Databricks fleet job.

Everything here is pure pandas/numpy on one region's rows at a time, with no Spark import,
so the same functions run under `applyInPandas` (databricks/04_train_fleet.py) and in a
plain local loop (scripts/backtest_local.py). Each model exposes:

    fit_predict(df, fold) -> DataFrame[period_utc, hour_local, actual, quantiles]

where `df` is one region's full gold history and `quantiles` holds the 19 levels in TAUS
order. The caller labels the result with model/region/fold.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

FOLD_COLUMNS = ("train_start", "train_end", "test_start", "test_end")


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. Definitions live in workspace.gold.backtest_folds and are
    read from there by every model, so no two models can score different months."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def from_row(cls, row: Any) -> "Fold":
        """Build from a row of backtest_folds (an itertuples row, Series, or dict)."""
        get = row.get if isinstance(row, dict) else lambda c: getattr(row, c)
        return cls(int(get("fold")), *(pd.Timestamp(get(c)) for c in FOLD_COLUMNS))


def usable(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Rows a model may train or be scored on: real published demand, inputs present.

    CLAUDE.md section 7: has_gap false, target > 0, and the model's own inputs non-null.
    Models therefore score slightly different row sets, which is why comparisons are made
    on the intersection.
    """
    keep = ~df.has_gap & (df.demand_mw > 0) & df[required].notna().all(axis=1)
    return df[keep].sort_values("period_utc")
