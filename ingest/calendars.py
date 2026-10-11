"""Holiday flags. CLAUDE.md section 4 calls holidays a major demand driver.

Why they matter physically: a federal holiday suppresses commercial and industrial load
while shifting residential load later in the morning, so the shape of the day changes, not
just its level. A model with weekday/weekend seasonality but no holiday term will
systematically over-forecast Thanksgiving and the Fourth of July.

Two judgment calls, both visible rather than buried:

1. **Which subdivision.** Single-state balancing authorities get that state's calendar
   (so CISO picks up Cesar Chavez Day); multi-state ones — PJM, MISO, SWPP — get
   federal-only, because no single state's calendar represents a region spanning a dozen
   of them. Set in `Region.holiday_subdiv` in regions.py.

2. **Observed vs. actual date.** The `holidays` package returns the *observed* date, so
   July 4th falling on a Saturday flags the Friday. That matches load behavior — businesses
   close when people take the day off, not on the nominal date.

Not handled, and worth knowing: the days *around* major holidays (the Friday after
Thanksgiving, Christmas Eve) behave like holidays for load but are not federal holidays.
A `days_to_nearest_holiday` feature would capture that. Left out because it is a modeling
decision, not plumbing.
"""

from __future__ import annotations

import functools
from datetime import date, datetime, timedelta

import holidays

from ingest.regions import get
from ingest.timeutils import local_date


@functools.lru_cache(maxsize=64)
def _calendar(subdiv: str | None, years: tuple[int, ...]) -> holidays.HolidayBase:
    """Build (and cache) a holiday calendar.

    Cached because constructing one is not free and ingestion asks about the same
    region-years combination thousands of times in a loop.
    """
    return holidays.country_holidays("US", subdiv=subdiv, years=years, observed=True)


def holiday_calendar(region: str, years: list[int]) -> holidays.HolidayBase:
    """The holiday calendar for a region, over the given years."""
    return _calendar(get(region).holiday_subdiv, tuple(sorted(set(years))))


def is_holiday(moment: datetime, region: str) -> bool:
    """Whether a UTC instant falls on a holiday in the region's *local* calendar.

    The local date is what matters: 2024-07-04 03:00 UTC is still July 3rd in Los Angeles,
    and flagging it as the holiday would shift the effect a day earlier for every western
    region. This is the same off-by-one-day trap that section 4 warns about for DST.
    """
    local = local_date(moment, get(region).timezone)
    year = int(local[:4])
    return local in holiday_calendar(region, [year])


def holiday_name(moment: datetime, region: str) -> str | None:
    """The holiday's name, or None. Useful for regime breakouts in section 7 — Thanksgiving
    and Independence Day do not have the same load signature, so pooling all holidays into
    one indicator discards information."""
    local = local_date(moment, get(region).timezone)
    year = int(local[:4])
    return holiday_calendar(region, [year]).get(local)


MAJOR_HOLIDAYS = frozenset(
    {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving Day",
        "Christmas Day",
    }
)
"""Federal holidays on which most businesses close, so demand drops noticeably. The rest of
the federal and state calendar (MLK Day, Veterans Day, Cesar Chavez Day, commemorative days
like Susan B. Anthony Day) closes government offices and banks but little else."""


def major_holiday_dates(years: list[int]) -> set[date]:
    """Dates on which most businesses close, the same for every region.

    The six `MAJOR_HOLIDAYS` on their observed dates, plus the day after Thanksgiving.
    That Friday is not a federal holiday, but load behaves like one. Computed from the
    federal calendar rather than matched by name, because state calendars disagree on what
    to call it ("Day After Thanksgiving", "Friday After Thanksgiving") or omit it entirely.
    """
    federal = _calendar(None, tuple(sorted(set(years))))
    out: set[date] = set()
    for day, name in federal.items():
        base = name.removesuffix(" (observed)")
        if base in MAJOR_HOLIDAYS:
            out.add(day)
        if base == "Thanksgiving Day":
            out.add(day + timedelta(days=1))
    return out


def holiday_dates(region: str, years: list[int]) -> dict[date, str]:
    """Every holiday in the window as {date: name}.

    Returned as a plain dict so it can be broadcast to Spark executors, which is the usual
    way to apply this without a Python UDF per row.
    """
    return dict(holiday_calendar(region, years).items())
