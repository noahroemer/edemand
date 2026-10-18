"""Scoring, defined once so every model is judged identically.

These definitions mirror databricks/06_backtest.py exactly (CLAUDE.md section 7): 19
quantiles per hour, CRPS approximated as 2 x mean pinball loss over them, and the same
metric names, so local results and Delta results can sit in the same table.
"""

import numpy as np
import pandas as pd

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05, 0.10, ..., 0.95
I10, I50, I90 = 1, 9, 17  # positions of the served quantiles in TAUS


def metrics(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """Score one group. `quantiles` is (n_hours, 19), columns ordered as TAUS."""
    diff = actual[:, None] - quantiles
    # Pinball loss: under-prediction costs tau, over-prediction costs (1 - tau).
    pinball = np.where(diff >= 0, TAUS * diff, (TAUS - 1) * diff)
    q10, q50, q90 = quantiles[:, I10], quantiles[:, I50], quantiles[:, I90]
    return {
        "n": float(len(actual)),
        "crps": 2 * pinball.mean(),
        "pinball_served": pinball[:, [I10, I50, I90]].mean(),
        "coverage80": 100 * np.mean((actual >= q10) & (actual <= q90)),
        "width80": float(np.mean(q90 - q10)),
        "mape": 100 * np.mean(np.abs(actual - q50) / actual),
        "bias": float(np.mean(actual - q50)),
    }


def metrics_by(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Score a predictions frame grouped by `by`. Needs `actual` and `quantiles` columns."""
    rows = []
    for keys, group in preds.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        scores = metrics(group.actual.to_numpy(), np.stack(group.quantiles.to_numpy()))
        rows.append(dict(zip(by, keys, strict=True)) | scores)
    return pd.DataFrame(rows)
