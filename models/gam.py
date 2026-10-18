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
is learned from genuinely out-of-sample residuals, then the model is refit for forecasting.

    fit          train_start .. train_end - 365 days  -> held-out model
    calibration  the last 365 days of training         -> z quantiles of its residuals
    refit        train_start .. train_end              -> the model that forecasts the test month
    test         q_tau = yhat + sigma * z_tau

Chosen by scripts/eda_gam_coverage.py (CLAUDE.md section 7): a 90-day window learned one
season's error shape and covered 76%; 365 days without refitting covered 82% because the
held-out model was a year stale; together they cover 80.0% with the best CRPS of six variants.
"""

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s, te

from models import Fold, usable
from models.intervals import quantiles_from, rolling_sigma, z_quantiles
from models.scoring import TAUS

FEATURES = ["temp_c", "hour_of_week", "day_of_year", "is_major_holiday", "lag_168h", "lag_48h"]
# heat=True adds a temperature x hour-of-day surface and a trailing 24h mean temperature.
HEAT_FEATURES = [*FEATURES, "hour_local", "temp_mean_24h"]
CALIBRATION_DAYS = 365  # a full year, so the z shape sees every season


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three derived columns the spline terms need."""
    out = df.copy()
    out["hour_of_week"] = out.dow * 24 + out.hour_local
    out["day_of_year"] = pd.to_datetime(out.local_date).dt.dayofyear
    out["is_major_holiday"] = out.is_major_holiday.astype(int)
    out = out.sort_values("period_utc")
    out["temp_mean_24h"] = out.temp_c.rolling(24, min_periods=1).mean()  # heat build-up
    return out


def specification(heat: bool = False) -> LinearGAM:
    """The model spec, unfitted. Term order matches FEATURES (or HEAT_FEATURES)."""
    terms = (
        s(0, n_splines=15)  # temp_c
        + s(1, n_splines=60, basis="cp", edge_knots=[0, 168])  # hour_of_week, wraps 167 -> 0
        + s(2, n_splines=12, basis="cp", edge_knots=[1, 367])  # day_of_year, wraps Dec -> Jan
        + f(3)  # is_major_holiday
        + l(4)  # lag_168h
        + l(5)  # lag_48h
    )
    if heat:
        # Cooling load lands in the afternoon and evening: let temperature's effect vary by
        # hour of day. And a hot day after hot days draws more than its own temperature says.
        terms = terms + te(0, 6, n_splines=[8, 8]) + s(7, n_splines=10)
    return LinearGAM(terms)


def fit(
    df: pd.DataFrame, fold: Fold, calibration_days: int = CALIBRATION_DAYS, heat: bool = False
) -> tuple[LinearGAM, pd.DataFrame]:
    """Fit on everything up to `calibration_days` before train_end. Returns the model and the
    usable rows (designed and filtered), so callers can plot terms or build intervals."""
    features = HEAT_FEATURES if heat else FEATURES
    rows = usable(design(df), features)  # design first: several features are derived
    calib_start = fold.train_end - pd.Timedelta(days=calibration_days)
    window = rows[(rows.period_utc >= fold.train_start) & (rows.period_utc < calib_start)]
    gam = specification(heat).fit(window[features].to_numpy(), window.demand_mw.to_numpy())
    return gam, rows


def fit_predict(
    df: pd.DataFrame,
    fold: Fold,
    taus: np.ndarray = TAUS,
    weather: str = "temp_c",
    calibration_days: int = CALIBRATION_DAYS,
    z_by_hour: bool = True,
    refit: bool = True,
    heat: bool = True,
) -> pd.DataFrame:
    """weather names the temperature column used at PREDICTION time. Training always uses
    temp_c (actual weather), as production would.

        "temp_c"      predict with actual weather too. Optimistic: at issue time tomorrow's
                      weather is only a forecast (CLAUDE.md section 7 leakage warning).
        "temp_c_pd1"  predict with the forecast issued a day earlier (Previous Runs API).
                      The honest number. Calibration residuals then carry forecast error as
                      well, so the intervals widen for it, as they would in production.
        "temp_c_hf"   Open-Meteo historical_forecast, kept to show it behaves like actual.

    Interval options (scripts/eda_gam_coverage.py compares them):

        calibration_days  length of the held-out window the z shape is learned on
        z_by_hour         learn z per local hour (24 tables) or pooled across hours
        refit             after calibrating, refit on the full training window and use that
                          fresher model for the test-month point forecast and residuals
        heat              temperature x hour-of-day surface plus trailing 24h mean temperature.
                          On by default: CRPS 1,256.7 -> 1,129.5 and heat-wave MAPE 5.47% ->
                          3.99% across 9 regions x 21 folds (CLAUDE.md section 7)
    """
    features = HEAT_FEATURES if heat else FEATURES
    gam, rows = fit(df, fold, calibration_days, heat)
    calib_start = fold.train_end - pd.Timedelta(days=calibration_days)

    # Predict calibration and test as one continuous series so sigma -- which needs the
    # 7 days ending 48h back -- is already available for the first test hour. The embargo
    # rows between train_end and test_start are not trained on, but their residuals are
    # known 48h later, which is exactly when sigma uses them.
    series = rows[(rows.period_utc >= calib_start) & (rows.period_utc < fold.test_end)].copy()
    if weather != "temp_c":
        series["temp_c"] = series[weather]
        series = series.dropna(subset=["temp_c"])
        series["temp_mean_24h"] = series.temp_c.rolling(24, min_periods=1).mean()
    series["yhat"] = gam.predict(series[features].to_numpy())
    series["error"] = series.demand_mw - series.yhat
    series["sigma"] = rolling_sigma(series.period_utc, series.error.abs())
    series = series[series.sigma > 0]

    calib = series[series.period_utc < fold.train_end]
    hour_key = calib.hour_local if z_by_hour else pd.Series(0, index=calib.index)
    z_table = z_quantiles(hour_key, calib.error / calib.sigma, taus)
    if not z_by_hour:  # one shape for every hour
        z_table = pd.DataFrame([z_table.loc[0].to_numpy()] * 24, columns=z_table.columns)

    if refit:
        # Point forecast from a model trained right up to train_end. Its residuals feed sigma
        # for the test month; the z shape still comes from the held-out calibration above.
        full = rows[(rows.period_utc >= fold.train_start) & (rows.period_utc < fold.train_end)]
        gam = specification(heat).fit(full[features].to_numpy(), full.demand_mw.to_numpy())
        series["yhat"] = gam.predict(series[features].to_numpy())
        series["error"] = series.demand_mw - series.yhat
        series["sigma"] = rolling_sigma(series.period_utc, series.error.abs())
        series = series[series.sigma > 0]

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
