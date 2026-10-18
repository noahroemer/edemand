"""Seasonal-naive benchmark: last week's value at the same hour, same UTC offset.

The local twin of databricks/06_backtest.py. It exists for two reasons: the Delta export
only carries q10/q50/q90, so CRPS can't be compared locally without it, and running both
implementations on the same rows is a check that the Spark one is right.

Never reported as a modelling result (CLAUDE.md section 2) -- it is the reference line
every other model has to beat.
"""

import numpy as np
import pandas as pd

from models import Fold, usable
from models.intervals import quantiles_from, rolling_sigma, z_quantiles
from models.scoring import TAUS

REQUIRED = ["lag_168h"]


def fit_predict(df: pd.DataFrame, fold: Fold, taus: np.ndarray = TAUS) -> pd.DataFrame:
    rows = usable(df, REQUIRED).copy()
    rows["center"] = rows.lag_168h
    rows["error"] = rows.demand_mw - rows.center
    rows["sigma"] = rolling_sigma(rows.period_utc, rows.error.abs())
    rows = rows[rows.sigma > 0]

    train = rows[(rows.period_utc >= fold.train_start) & (rows.period_utc < fold.train_end)]
    z_table = z_quantiles(train.hour_local, train.error / train.sigma, taus)

    test = rows[(rows.period_utc >= fold.test_start) & (rows.period_utc < fold.test_end)]
    q = quantiles_from(test.center.to_numpy(), test.sigma.to_numpy(), test.hour_local, z_table)
    return pd.DataFrame(
        {
            "period_utc": test.period_utc.to_numpy(),
            "hour_local": test.hour_local.to_numpy(),
            "actual": test.demand_mw.to_numpy(),
            "quantiles": list(q),
        }
    )
