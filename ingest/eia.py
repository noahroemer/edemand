"""EIA API v2 client for hourly demand by balancing authority.

Route: `electricity/rto/region-data`, which EIA documents with two frequencies —
`hourly` (UTC) and `local-hourly` (wall-clock local). **We use `hourly` exclusively**, per
CLAUDE.md section 4's store-everything-in-UTC rule. Facets are `respondent` (the BA code)
and `type` (the metric); `D` is demand.

Three things about this API that shape the code:

1. **5000 rows per JSON request, maximum.** A year of hourly data for one region is ~8760
   rows, so a single year needs two requests. Pagination is via `offset` and `length`.

2. **Pagination without an explicit sort is unsafe.** `offset` is only meaningful against a
   stable ordering. Without `sort`, rows can be duplicated or skipped across pages. Every
   request here sorts by `period` ascending.

3. **EIA revises published values.** A row fetched today can differ from the same row
   fetched last week. That is why section 4 mandates Delta MERGE upserts keyed on
   (region, period_utc) rather than appends — this client therefore returns exactly that
   key plus the value, and records `fetched_at` so revisions are traceable.

Returns `list[dict]`, not a DataFrame, on purpose: CLAUDE.md section 12 says write PySpark
first rather than prototyping in pandas. Plain records hand straight to
`spark.createDataFrame(...)` without pandas in the middle.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ingest.timeutils import format_eia_period, parse_eia_period, require_utc

log = logging.getLogger(__name__)

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
"""Balancing-authority level: facets `respondent` and `type`."""

SUBBA_URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
"""Subregion level: facets `parent` and `subba`, demand only, no `type`. The CAISO
sub-areas in CLAUDE.md section 4 are only available here - `region-data` returns an empty
result for them, not an error."""

MAX_ROWS_PER_REQUEST = 5000
"""EIA's documented JSON ceiling. Requesting more is silently truncated, which would look
like missing data rather than an error."""

DEMAND = "D"
DAY_AHEAD_FORECAST = "DF"
"""EIA's own day-ahead forecast. Worth ingesting as a free extra benchmark: beating the
operator's published forecast is a more meaningful claim than beating seasonal-naive."""


class EIAError(RuntimeError):
    """Request failed, or the response envelope was not shaped as documented."""


def _session(retries: int = 4) -> requests.Session:
    """Session with backoff on the status codes worth retrying.

    429 and 5xx are transient; 403 (bad key) and 400 (bad parameters) are not, and retrying
    them just delays a clear error. `respect_retry_after_header` honors EIA's own throttle
    guidance instead of guessing.
    """
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


def resolve_api_key(explicit: str | None = None) -> str:
    """Return the API key, preferring an explicit argument over EIA_API_KEY.

    Strips surrounding quotes and whitespace. `.env` files do not require quoting, but
    pasting a quoted value is an easy mistake and the symptom is a bare 403 that looks
    like an invalid key rather than a formatting problem. python-dotenv strips quotes on
    its own; this covers the case where the variable arrives from somewhere that does not.
    """
    key = explicit or os.environ.get("EIA_API_KEY")
    if key:
        key = key.strip().strip("\"'").strip()
    if not key:
        raise EIAError(
            "No EIA API key. Set EIA_API_KEY in .env or pass api_key=. "
            "Register free at https://www.eia.gov/opendata/register.php"
        )
    return key


def _coerce_value(raw: Any) -> float | None:
    """EIA returns numbers as JSON strings, and genuinely missing hours as null or ''.

    Missing stays None. It must not become 0.0 — zero demand is a physically meaningful
    value (and a real event during outages), so conflating the two would corrupt both the
    gap flag and any model trained on it.
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("Non-numeric demand value %r; treating as missing", raw)
        return None


def fetch_demand(
    region: str,
    start: datetime,
    end: datetime,
    *,
    api_key: str | None = None,
    data_type: str = DEMAND,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    """Fetch hourly demand for one balancing authority over [start, end], inclusive.

    `start` and `end` must be aware datetimes; naive ones are rejected by `require_utc`
    rather than being assumed to mean UTC.

    Returns records keyed for the (region, period_utc) MERGE:

        {"region": "CISO", "period_utc": datetime(..., tzinfo=UTC),
         "demand_mw": 24913.0, "data_type": "D",
         "fetched_at": datetime(..., tzinfo=UTC)}

    Rows are sorted by period ascending and deduplicated on period — EIA has been observed
    to return the same period twice across page boundaries, and taking the last occurrence
    keeps the behavior deterministic.
    """
    key = resolve_api_key(api_key)
    first = require_utc(start)
    last = require_utc(end)
    own_session = session is None
    http = session or _session()

    params: dict[str, Any] = {
        "api_key": key,
        "frequency": "hourly",  # UTC. 'local-hourly' would violate section 4.
        "data[0]": "value",
        "facets[respondent][]": region.upper(),
        "facets[type][]": data_type,
        "start": format_eia_period(first),
        "end": format_eia_period(last),
        # Stable ordering is what makes offset pagination correct.
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": MAX_ROWS_PER_REQUEST,
    }

    # One timestamp for the whole fetch, so every row from this call shares a revision
    # stamp. Per-row timestamps would imply the rows were observed at different times.
    fetched_at = datetime.now(UTC)
    by_period: dict[datetime, dict[str, Any]] = {}
    offset = 0
    total: int | None = None

    try:
        for page in range(max_pages):
            payload = _request(http, {**params, "offset": offset}, timeout)
            rows = payload.get("data")
            if rows is None:
                raise EIAError(f"Response had no 'data' key: {sorted(payload)}")

            if total is None:
                # EIA returns total as a string in some responses and an int in others.
                total = int(payload.get("total", len(rows)) or 0)
                log.info("%s %s: %d rows available", region, data_type, total)

            for row in rows:
                period = parse_eia_period(row["period"])
                by_period[period] = {
                    "region": region.upper(),
                    "period_utc": period,
                    "demand_mw": _coerce_value(row.get("value")),
                    "data_type": data_type,
                    "fetched_at": fetched_at,
                }

            offset += len(rows)
            # Stop on a short page as well as on the count: a short page means the server
            # has no more rows, regardless of what `total` claimed.
            if not rows or offset >= total:
                break
            if page == max_pages - 1:
                raise EIAError(
                    f"Hit max_pages={max_pages} with {offset}/{total} rows for {region}. "
                    "Narrow the window or raise max_pages."
                )
    finally:
        if own_session:
            http.close()

    return [by_period[p] for p in sorted(by_period)]


def _request(
    http: requests.Session,
    params: dict[str, Any],
    timeout: float,
    url: str = BASE_URL,
) -> dict[str, Any]:
    """One request. Unwraps EIA's `{"response": {...}}` envelope and keeps the key out of logs."""
    try:
        resp = http.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise EIAError(f"Request to EIA failed: {type(exc).__name__}: {exc}") from exc

    if resp.status_code == 403:
        raise EIAError("EIA returned 403. The API key is missing, invalid, or revoked.")
    if resp.status_code == 400:
        # EIA puts useful parameter diagnostics in the body here.
        raise EIAError(f"EIA returned 400 (bad request): {resp.text[:400]}")
    if not resp.ok:
        raise EIAError(f"EIA returned {resp.status_code}: {resp.text[:400]}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise EIAError(f"EIA returned non-JSON: {resp.text[:200]}") from exc

    if "response" not in body:
        # A warnings-only body usually means the facets matched nothing.
        raise EIAError(f"Unexpected envelope, keys={sorted(body)}: {str(body)[:300]}")
    return body["response"]


def fetch_subba_demand(
    parent: str,
    start: datetime,
    end: datetime,
    *,
    subbas: list[str] | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    """Fetch hourly demand for a parent BA's **subregions**.

    Separate route from `fetch_demand`, which is the discovery that matters here: the
    `region-data` route returns nothing for PGAE/SCE/SDGE/VEA, because sub-balancing
    authorities live on `region-sub-ba-data` with facets `parent` and `subba` instead of
    `respondent` and `type`. Querying the wrong route returns an empty result rather than
    an error, which would look like "CAISO sub-areas have no data" and quietly undermine
    the hierarchical model in CLAUDE.md section 6.

    There is no `type` facet here - the route publishes demand only, so no DF benchmark
    exists at subregion level.

    Deduplication is on (subba, period_utc), not period alone: one period carries one row
    per subregion, so deduplicating on period would discard three of every four rows.

    Records carry `parent_region` so the hierarchy is explicit in the data, which is what
    MinT reconciliation needs to build its summation matrix.
    """
    key = resolve_api_key(api_key)
    first = require_utc(start)
    last = require_utc(end)
    own_session = session is None
    http = session or _session()

    params: dict[str, Any] = {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[parent][]": parent.upper(),
        "start": format_eia_period(first),
        "end": format_eia_period(last),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": MAX_ROWS_PER_REQUEST,
    }
    if subbas:
        for i, code in enumerate(subbas):
            params[f"facets[subba][{i}]"] = code.upper()

    fetched_at = datetime.now(UTC)
    by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    offset = 0
    total: int | None = None

    try:
        for page in range(max_pages):
            payload = _request(http, {**params, "offset": offset}, timeout, url=SUBBA_URL)
            rows = payload.get("data")
            if rows is None:
                raise EIAError(f"Response had no 'data' key: {sorted(payload)}")
            if total is None:
                total = int(payload.get("total", len(rows)) or 0)
                log.info("%s subregions: %d rows available", parent, total)

            for row in rows:
                period = parse_eia_period(row["period"])
                subba = row["subba"]
                by_key[(subba, period)] = {
                    "region": subba,
                    "parent_region": parent.upper(),
                    "period_utc": period,
                    "demand_mw": _coerce_value(row.get("value")),
                    "data_type": DEMAND,
                    "fetched_at": fetched_at,
                }

            offset += len(rows)
            if not rows or offset >= total:
                break
            if page == max_pages - 1:
                raise EIAError(
                    f"Hit max_pages={max_pages} with {offset}/{total} rows for {parent} subregions."
                )
    finally:
        if own_session:
            http.close()

    return [by_key[k] for k in sorted(by_key, key=lambda k: (k[1], k[0]))]


def fetch_many(
    region_codes: list[str],
    start: datetime,
    end: datetime,
    *,
    api_key: str | None = None,
    data_type: str = DEMAND,
) -> list[dict[str, Any]]:
    """Fetch several regions over one connection.

    Sequential, not concurrent: EIA throttles per key, and ten regions of a few pages each
    is not worth the complexity or the risk of a 429 storm. One Session means one TLS
    handshake for all of it.
    """
    key = resolve_api_key(api_key)
    out: list[dict[str, Any]] = []
    with _session() as http:
        for code in region_codes:
            out.extend(
                fetch_demand(code, start, end, api_key=key, data_type=data_type, session=http)
            )
    return out
