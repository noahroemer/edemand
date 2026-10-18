"""Volatility-scaled prediction intervals: scale x shape (CLAUDE.md section 7a).

    q_tau(t) = center(t) + sigma(t) * z_tau(hour_local)

    center    the point forecast (lag_168h for seasonal-naive, yhat for the GAM)
    sigma(t)  mean |error| over the 7 days ending 48h before t -- the *scale*. Known at
              issue time, so it is computed over train and test rows alike.
    z_tau     quantiles of the standardised error e/sigma -- the *shape*, learned on
              training rows only, separately per local hour.

Why split it that way: measured on seasonal-naive, raw error spread grew 17% from train
to test (demand growth), while the standardised z spread moved 0.6%. Scale drifts with
recent conditions; shape does not. Scaling cut mean |coverage - 80| from 8.85 points to
2.35 across 21 folds. The same idea is normalised conformal prediction, and it is the
plan for every model here.
"""

import numpy as np
import pandas as pd

EMBARGO_HOURS = 48  # nothing newer than this is known when the forecast is issued
SIGMA_WINDOW_DAYS = 7


def rolling_sigma(
    period_utc: pd.Series,
    abs_error: pd.Series,
    embargo_hours: int = EMBARGO_HOURS,
    window_days: int = SIGMA_WINDOW_DAYS,
) -> pd.Series:
    """Mean |error| over the `window_days` ending `embargo_hours` before each row.

    Implemented by shifting the errors *forward* in time and then taking a trailing
    window, which is the same thing and keeps the embargo impossible to get wrong.
    """
    series = pd.Series(abs_error.to_numpy(), index=pd.DatetimeIndex(period_utc))
    shifted = series.shift(freq=f"{embargo_hours}h")
    # Union the indexes first: a row whose t - 48h is missing still needs a sigma, and
    # rolling() skips the NaN that union introduces rather than being poisoned by it.
    full = shifted.reindex(shifted.index.union(series.index))
    sigma = full.rolling(f"{window_days * 24}h").mean().reindex(series.index)
    return pd.Series(sigma.to_numpy(), index=abs_error.index)


def z_quantiles(hour_local: pd.Series, z: pd.Series, taus: np.ndarray) -> pd.DataFrame:
    """Quantiles of the standardised error per local hour: index hour_local, columns taus.

    Error is far from uniform across the day -- CISO's hardest hour (17, solar collapsing)
    runs ~2.5x the easiest (5-7am) -- so one set of offsets for all hours is miscalibrated
    at both ends.
    """
    frame = pd.DataFrame({"hour_local": hour_local.to_numpy(), "z": z.to_numpy()})
    table = frame.groupby("hour_local").z.quantile(taus).unstack()
    # An hour absent from training (possible only in a pathological fold) borrows its
    # neighbours' shape rather than crashing the fit.
    return table.reindex(range(24)).ffill().bfill()


def quantiles_from(
    center: np.ndarray, sigma: np.ndarray, hour_local: pd.Series, z_table: pd.DataFrame
) -> np.ndarray:
    """(n_hours, n_taus) quantile matrix. Rows cannot cross: sigma > 0 and z is sorted."""
    return center[:, None] + sigma[:, None] * z_table.loc[hour_local].to_numpy()
