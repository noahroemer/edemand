"""Open-Meteo weather client. No API key required.

Three endpoints, and **choosing between them is a modeling decision, not a plumbing
detail** — CLAUDE.md section 7's leakage warning is exactly about this:

| Function | Endpoint | Returns |
| --- | --- | --- |
| `fetch_archive` | `archive-api` | ERA5 reanalysis — best estimate of what the weather *was*. Back to 1940. |
| `fetch_historical_forecast` | `historical-forecast-api` | What the forecast *said it would be*, as issued. From ~2022. |
| `fetch_forecast` | `api` | The live forward forecast, for serving. |

A backtest trained and scored on `fetch_archive` is **optimistic**: at 06:00 serving time
you have a forecast, not a reanalysis, and forecast temperature carries error that
reanalysis does not. `fetch_historical_forecast` is the honest counterpart — it is the same
kind of data the model will actually see in production, which is why it exists here.

ERA5 also lags real time by about 5 days, so it cannot feed a live daily job even if you
wanted it to. The archive is for long history; archived forecasts are for honest evaluation.

Two practical notes:
  - `timezone=GMT` is requested explicitly. Open-Meteo defaults to GMT, but relying on a
    default for the one thing CLAUDE.md section 4 is strictest about is a bad trade.
  - Multiple coordinates can be batched into a single request, and the response becomes a
    JSON array in the order requested. Ten regions in one call instead of ten.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
)
"""Temperature is the dominant weather driver of electricity demand; humidity matters
because cooling load tracks apparent temperature rather than dry-bulb. Both are listed in
the gold.features schema in CLAUDE.md section 9."""

# Open-Meteo names variables in its response exactly as requested; these are our column
# names, kept separate so the schema does not inherit the vendor's naming.
COLUMN_NAMES = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    # Previous Runs API: same variables, as forecast N days before the hour.
    "temperature_2m_previous_day1": "temp_c",
    "relative_humidity_2m_previous_day1": "humidity_pct",
}

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ARCHIVE_LAG_DAYS = 5
"""ERA5 reanalysis lags real time by roughly this much. Requesting more recent dates
returns fewer rows than asked for rather than an error."""


class OpenMeteoError(RuntimeError):
    """Request failed, or the response was not shaped as documented."""


def _session(retries: int = 4) -> requests.Session:
    session = requests.Session()
    policy = Retry(
        total=retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=policy))
    return session


def _parse_hourly_block(
    block: dict[str, Any],
    variables: tuple[str, ...],
    latitude: float,
    longitude: float,
    source: str,
) -> list[dict[str, Any]]:
    """Turn Open-Meteo's column-oriented hourly block into row records.

    The API returns parallel arrays (`time`, `temperature_2m`, ...). Row orientation is what
    Spark wants, and it makes a missing value at index i obvious rather than requiring
    cross-referencing two lists.

    Times come back as "2024-01-01T00:00" with no offset. Because we requested
    `timezone=GMT`, these are UTC, and they are tagged as such here rather than anywhere
    downstream.
    """
    hourly = block.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise OpenMeteoError(f"Response had no hourly.time block: {sorted(block)}")

    times = hourly["time"]
    series = {var: hourly.get(var, [None] * len(times)) for var in variables}
    fetched_at = datetime.now(UTC)

    rows: list[dict[str, Any]] = []
    for i, stamp in enumerate(times):
        record: dict[str, Any] = {
            "period_utc": datetime.fromisoformat(stamp).replace(tzinfo=UTC),
            "latitude": latitude,
            "longitude": longitude,
            "source": source,
            "fetched_at": fetched_at,
        }
        for var in variables:
            values = series[var]
            record[COLUMN_NAMES.get(var, var)] = values[i] if i < len(values) else None
        rows.append(record)
    return rows


def _request(
    url: str,
    params: dict[str, Any],
    *,
    session: requests.Session | None,
    timeout: float,
) -> list[dict[str, Any]]:
    """One request. Normalizes the single-location object and the multi-location array
    into a list, so callers have one shape to handle."""
    own = session is None
    http = session or _session()
    try:
        try:
            resp = http.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise OpenMeteoError(f"Request to {url} failed: {type(exc).__name__}: {exc}") from exc

        if not resp.ok:
            # Open-Meteo returns {"error": true, "reason": "..."} with useful detail.
            raise OpenMeteoError(f"{url} returned {resp.status_code}: {resp.text[:400]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise OpenMeteoError(f"{url} returned non-JSON: {resp.text[:200]}") from exc
    finally:
        if own:
            http.close()

    if isinstance(body, dict) and body.get("error"):
        raise OpenMeteoError(f"Open-Meteo error: {body.get('reason', body)}")
    return body if isinstance(body, list) else [body]


def _fetch(
    url: str,
    coordinates: list[tuple[float, float]],
    params: dict[str, Any],
    variables: tuple[str, ...],
    source: str,
    *,
    session: requests.Session | None,
    timeout: float,
) -> list[dict[str, Any]]:
    if not coordinates:
        raise ValueError("No coordinates given")
    full = {
        **params,
        # Batched as comma-separated lists; response order matches request order.
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in coordinates),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in coordinates),
        "hourly": ",".join(variables),
        "timezone": "GMT",
    }
    blocks = _request(url, full, session=session, timeout=timeout)
    if len(blocks) != len(coordinates):
        raise OpenMeteoError(
            f"Asked for {len(coordinates)} locations, got {len(blocks)} blocks back"
        )
    rows: list[dict[str, Any]] = []
    for (lat, lon), block in zip(coordinates, blocks, strict=True):
        rows.extend(_parse_hourly_block(block, variables, lat, lon, source))
    return rows


def fetch_archive(
    coordinates: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """ERA5 reanalysis for [start_date, end_date] (YYYY-MM-DD, inclusive).

    Use for long training history. **Do not report a backtest scored on this as if it were
    achievable live** — see the module docstring and CLAUDE.md section 7. Lags real time by
    about `ARCHIVE_LAG_DAYS`.
    """
    return _fetch(
        ARCHIVE_URL,
        coordinates,
        {"start_date": start_date, "end_date": end_date},
        variables,
        source="era5_archive",
        session=session,
        timeout=timeout,
    )


def fetch_historical_forecast(
    coordinates: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Archived operational forecasts, stitched from the start of each model run.

    **Not a day-ahead forecast.** Measured in the backtest (CLAUDE.md section 7): a GAM
    predicting with this data scores the same as with actual weather (CRPS 1,223.7 vs
    1,224.8), so its lead time is far shorter than the 11-34h our forecasts need. Use
    `fetch_previous_day` to measure weather-forecast leakage instead.
    """
    return _fetch(
        HISTORICAL_FORECAST_URL,
        coordinates,
        {"start_date": start_date, "end_date": end_date},
        variables,
        source="historical_forecast",
        session=session,
        timeout=timeout,
    )


def fetch_previous_day(
    coordinates: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """What the forecast said one day earlier, from Open-Meteo's Previous Runs API.

    Each hour's value comes from the model run issued about 24h before the run that would
    normally cover it, so the lead time is roughly 24-47h. Our day-ahead forecast (issued
    13:00 UTC for the next UTC day) needs 11-34h, so this is slightly conservative: if
    anything it overstates weather-forecast error. This is the honest evaluation input.
    Coverage goes back to 2023 with a few percent of hours missing (more in 2024).
    """
    return _fetch(
        PREVIOUS_RUNS_URL,
        coordinates,
        {"start_date": start_date, "end_date": end_date},
        ("temperature_2m_previous_day1", "relative_humidity_2m_previous_day1"),
        source="previous_day1",
        session=session,
        timeout=timeout,
    )


def fetch_forecast(
    coordinates: list[tuple[float, float]],
    *,
    forecast_days: int = 2,
    past_days: int = 0,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """The live forward forecast, for the daily serving job.

    `forecast_days=2` rather than 1: the 24-hour horizon is issued at 06:00 local
    (section 10) and therefore extends into the following calendar day, so one day of
    forecast would run out partway through the horizon.
    """
    return _fetch(
        FORECAST_URL,
        coordinates,
        {"forecast_days": forecast_days, "past_days": past_days},
        variables,
        source="live_forecast",
        session=session,
        timeout=timeout,
    )
