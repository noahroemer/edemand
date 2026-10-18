"""Generalised additive model: demand as a sum of smooth curves (CLAUDE.md section 6).

    s(temp_c)                 temperature response, learned -- expect a U shape
    s(hour_of_week, cyclic)   0..167, so Saturday's daily shape can differ from Tuesday's
    s(day_of_year, cyclic)    season, wrapping December into January
    f(is_major_holiday)       a level shift on major holidays
    l(lag_168h), l(lag_48h)   recent demand level

Each smooth is a penalised spline: flexible enough to bend where the data bends, penalised
so it does not chase noise. Unlike a boosted tree it is readable -- the temperature partial
dependence *is* the U-shaped response curve -- and unlike seasonal-naive it can respond to
tomorrow's weather instead of repeating last week.

Intervals reuse the volatility-scaled method from models/intervals.py, with one wrinkle:
in-sample residuals are too small, so the training window is split and the interval shape
is learned from genuinely out-of-sample residuals.

    fit          train_start .. train_end - 90 days
    calibration  the last 90 days of training -> z quantiles
    test         the fold's test month        -> q_tau = yhat + sigma * z_tau
"""

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s

from models import Fold, usable
from models.intervals import quantiles_from, rolling_sigma, z_quantiles
from models.scoring import TAUS

FEATURES = ["temp_c", "hour_of_week", "day_of_year", "is_major_holiday", "lag_168h", "lag_48h"]
CALIBRATION_DAYS = 90


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three derived columns the spline terms need."""
    out = df.copy()
    out["hour_of_week"] = out.dow * 24 + out.hour_local
    out["day_of_year"] = pd.to_datetime(out.local_date).dt.dayofyear
    out["is_major_holiday"] = out.is_major_holiday.astype(int)
    return out


def specification() -> LinearGAM:
    """The model spec, unfitted. Term order matches FEATURES."""
    return LinearGAM(
        s(0, n_splines=15)  # temp_c
        + s(1, n_splines=60, basis="cp", edge_knots=[0, 168])  # hour_of_week, wraps 167 -> 0
        + s(2, n_splines=12, basis="cp", edge_knots=[1, 367])  # day_of_year, wraps Dec -> Jan
        + f(3)  # is_major_holiday
        + l(4)  # lag_168h
        + l(5)  # lag_48h
    )


def fit(df: pd.DataFrame, fold: Fold) -> tuple[LinearGAM, pd.DataFrame]:
    """Fit on everything up to 90 days before train_end. Returns the model and the usable
    rows (designed and filtered), so callers can plot terms or build intervals from them."""
    rows = usable(design(df), FEATURES)  # design first: three of FEATURES are derived
    calib_start = fold.train_end - pd.Timedelta(days=CALIBRATION_DAYS)
    window = rows[(rows.period_utc >= fold.train_start) & (rows.period_utc < calib_start)]
    gam = specification().fit(window[FEATURES].to_numpy(), window.demand_mw.to_numpy())
    return gam, rows


def fit_predict(
    df: pd.DataFrame, fold: Fold, taus: np.ndarray = TAUS, weather: str = "archive"
) -> pd.DataFrame:
    """weather picks the temperature used at PREDICTION time. Training always uses archive
    (actual) weather, as production would.

        "archive"   predict with actual weather too. Optimistic: at issue time tomorrow's
                    weather is only a forecast (CLAUDE.md section 7 leakage warning).
        "forecast"  predict with the archived forecast (df needs a temp_c_fc column). The
                    calibration residuals then carry forecast error as well, so the
                    intervals widen for it, as they would in production.
    """
    gam, rows = fit(df, fold)
    calib_start = fold.train_end - pd.Timedelta(days=CALIBRATION_DAYS)

    # Predict calibration and test as one continuous series so sigma -- which needs the
    # 7 days ending 48h back -- is already available for the first test hour. The embargo
    # rows between train_end and test_start are not trained on, but their residuals are
    # known 48h later, which is exactly when sigma uses them.
    series = rows[(rows.period_utc >= calib_start) & (rows.period_utc < fold.test_end)].copy()
    if weather == "forecast":
        series["temp_c"] = series.temp_c_fc
        series = series.dropna(subset=["temp_c"])
    elif weather != "archive":
        raise ValueError(f"weather must be 'archive' or 'forecast', not {weather!r}")
    series["yhat"] = gam.predict(series[FEATURES].to_numpy())
    series["error"] = series.demand_mw - series.yhat
    series["sigma"] = rolling_sigma(series.period_utc, series.error.abs())
    series = series[series.sigma > 0]

    calib = series[series.period_utc < fold.train_end]
    z_table = z_quantiles(calib.hour_local, calib.error / calib.sigma, taus)

    test = series[series.period_utc >= fold.test_start]
    q = quantiles_from(test.yhat.to_numpy(), test.sigma.to_numpy(), test.hour_local, z_table)
    return pd.DataFrame(
        {
            "period_utc": test.period_utc.to_numpy(),
            "hour_local": test.hour_local.to_numpy(),
            "actual": test.demand_mw.to_numpy(),
            "quantiles": list(q),
        }
    )
