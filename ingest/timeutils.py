"""UTC and DST handling. CLAUDE.md section 4 calls this the number one source of bugs in
this domain, so the rules are enforced here rather than left to each caller.

The invariant: **every timestamp crossing a module boundary is timezone-aware UTC.**
Local time appears only as derived integer features (hour_local, dow), never as a stored
timestamp.

Why that works: converting UTC to local is *total* — every UTC instant has exactly one
local representation, including during DST transitions. Converting local to UTC is *not*:
on a spring-forward date 02:30 local does not exist, and on a fall-back date 01:30 local
happens twice. So the pipeline only ever goes UTC to local, never back. A local time zone
with a DST gap or fold cannot round-trip, which is how forecast systems silently lose or
duplicate an hour twice a year.

Consequences you can see in the data, and which tests/test_timeutils.py asserts:
  - a spring-forward local day has 23 distinct local hours
  - a fall-back local day has 25, with hour_local == 1 appearing twice
Both are correct. Neither is a gap. The gap detector works on UTC, where every day has 24
hours, so it is immune to this entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

# EIA v2 hourly periods look like "2024-03-10T07" - date, 'T', zero-padded UTC hour.
EIA_HOUR_FORMAT = "%Y-%m-%dT%H"


def parse_eia_period(period: str) -> datetime:
    """Parse an EIA v2 hourly period string into an aware UTC datetime.

    Only valid for `frequency=hourly`, which EIA documents as UTC. The `local-hourly`
    frequency returns wall-clock local time with an offset suffix and must not be fed
    here - that is the trap this function exists to prevent.
    """
    return datetime.strptime(period, EIA_HOUR_FORMAT).replace(tzinfo=UTC)


def format_eia_period(moment: datetime) -> str:
    """Format an aware datetime as an EIA hourly period for start/end parameters."""
    return require_utc(moment).strftime(EIA_HOUR_FORMAT)


def require_utc(moment: datetime) -> datetime:
    """Return `moment` as UTC, refusing naive datetimes.

    Rejecting rather than assuming is deliberate: a naive datetime silently interpreted as
    UTC is how local timestamps leak into a UTC-only store, and the resulting 7- or 8-hour
    offset looks like a plausible demand curve rather than an obvious error.
    """
    if moment.tzinfo is None:
        raise ValueError(
            f"Naive datetime {moment!r}. Pass an aware UTC datetime, "
            "e.g. datetime(2024, 1, 1, tzinfo=UTC)."
        )
    return moment.astimezone(UTC)


def hour_grid(start: datetime, end: datetime) -> list[datetime]:
    """Every hour from `start` to `end` inclusive, on the hour, in UTC.

    The reference grid for gap detection: what *should* be present. In UTC this is always
    exactly one row per hour, with no DST irregularity.
    """
    first = require_utc(start).replace(minute=0, second=0, microsecond=0)
    last = require_utc(end).replace(minute=0, second=0, microsecond=0)
    if last < first:
        raise ValueError(f"end {last.isoformat()} precedes start {first.isoformat()}")
    out: list[datetime] = []
    cursor = first
    while cursor <= last:
        out.append(cursor)
        cursor += timedelta(hours=1)
    return out


def find_gaps(periods: list[datetime], start: datetime, end: datetime) -> list[datetime]:
    """Hours in [start, end] that are missing from `periods`.

    Returns the missing hours rather than filling them. CLAUDE.md section 4: never
    silently forward-fill — flag and handle explicitly. The Spark layer decides what to do;
    this only reports what is absent.
    """
    present = {require_utc(p).replace(minute=0, second=0, microsecond=0) for p in periods}
    return [hour for hour in hour_grid(start, end) if hour not in present]


def local_calendar_features(moment: datetime, timezone: str) -> dict[str, int]:
    """Derive calendar features from a UTC instant, in the region's local time.

    Local time is where human behavior lives: people wake, work, and run air conditioning
    on local clocks, so seasonality must be indexed locally even though storage is UTC.

    Returns ints only — no local timestamp escapes this function.
    """
    local = require_utc(moment).astimezone(ZoneInfo(timezone))
    return {
        "hour_local": local.hour,
        "dow": local.weekday(),  # Monday = 0, matching pandas and Spark's dayofweek-1
        "month": local.month,
        "day_of_year": local.timetuple().tm_yday,
        "is_weekend": int(local.weekday() >= 5),
    }


def local_date(moment: datetime, timezone: str) -> str:
    """The local calendar date (YYYY-MM-DD) for a UTC instant.

    Needed for holiday lookup: a UTC timestamp near midnight belongs to a different local
    date than its UTC date, and getting that wrong misattributes holiday effects by a day.
    """
    return require_utc(moment).astimezone(ZoneInfo(timezone)).strftime("%Y-%m-%d")


def utc_offset_hours(moment: datetime, timezone: str) -> float:
    """The region's UTC offset at this instant, in hours (e.g. -7.0 for PDT).

    Useful as a diagnostic: plotting this over a year makes DST transitions visible, and a
    constant value where one is not expected means the timezone string is wrong.
    """
    local = require_utc(moment).astimezone(ZoneInfo(timezone))
    offset = local.utcoffset()
    if offset is None:  # pragma: no cover - ZoneInfo always supplies one
        raise ValueError(f"No UTC offset for {timezone}")
    return offset.total_seconds() / 3600.0
