"""DST and gap-handling tests. CLAUDE.md section 12 names both as required coverage.

The dates used are real US DST transitions:
    2024-03-10  spring forward, 02:00-02:59 local does not exist
    2024-11-03  fall back, 01:00-01:59 local happens twice
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ingest.timeutils import (
    find_gaps,
    format_eia_period,
    hour_grid,
    local_calendar_features,
    local_date,
    parse_eia_period,
    require_utc,
    utc_offset_hours,
)

LA = "America/Los_Angeles"
NY = "America/New_York"


class TestUtcDiscipline:
    def test_naive_datetime_is_rejected(self):
        """Assuming naive means UTC is how local timestamps leak into a UTC store."""
        with pytest.raises(ValueError, match="Naive datetime"):
            require_utc(datetime(2024, 1, 1, 12))  # noqa: DTZ001 - the point of the test

    def test_non_utc_aware_datetime_is_converted(self):
        from zoneinfo import ZoneInfo

        noon_la = datetime(2024, 1, 1, 12, tzinfo=ZoneInfo(LA))
        assert require_utc(noon_la) == datetime(2024, 1, 1, 20, tzinfo=UTC)

    def test_eia_period_round_trip(self):
        period = "2024-03-10T07"
        parsed = parse_eia_period(period)
        assert parsed == datetime(2024, 3, 10, 7, tzinfo=UTC)
        assert format_eia_period(parsed) == period


class TestSpringForward:
    """2024-03-10: local clocks jump 02:00 -> 03:00. The local day has 23 hours."""

    def test_local_day_has_23_distinct_hours(self):
        # 2024-03-10 08:00Z is midnight local (PST, UTC-8). 23 hours covers the local day.
        start = datetime(2024, 3, 10, 8, tzinfo=UTC)
        hours = {
            local_calendar_features(start + timedelta(hours=i), LA)["hour_local"] for i in range(23)
        }
        assert len(hours) == 23
        assert 2 not in hours, "02:00 local does not exist on a spring-forward date"

    def test_offset_shifts_mid_day(self):
        before = datetime(2024, 3, 10, 9, tzinfo=UTC)  # 01:00 PST
        after = datetime(2024, 3, 10, 11, tzinfo=UTC)  # 04:00 PDT
        assert utc_offset_hours(before, LA) == -8.0
        assert utc_offset_hours(after, LA) == -7.0

    def test_utc_grid_is_unaffected(self):
        """The whole point of storing UTC: no irregularity to handle."""
        grid = hour_grid(
            datetime(2024, 3, 10, 0, tzinfo=UTC), datetime(2024, 3, 10, 23, tzinfo=UTC)
        )
        assert len(grid) == 24


class TestFallBack:
    """2024-11-03: local clocks repeat 01:00-01:59. The local day has 25 hours."""

    def test_hour_one_appears_twice(self):
        # 2024-11-03 08:00Z = 01:00 PDT; 09:00Z = 01:00 PST. Same local hour, distinct instants.
        first = datetime(2024, 11, 3, 8, tzinfo=UTC)
        second = datetime(2024, 11, 3, 9, tzinfo=UTC)
        assert local_calendar_features(first, LA)["hour_local"] == 1
        assert local_calendar_features(second, LA)["hour_local"] == 1
        assert first != second
        assert utc_offset_hours(first, LA) == -7.0
        assert utc_offset_hours(second, LA) == -8.0

    def test_local_day_spans_25_utc_hours(self):
        start = datetime(2024, 11, 3, 7, tzinfo=UTC)  # midnight PDT
        dates = {local_date(start + timedelta(hours=i), LA) for i in range(25)}
        assert dates == {"2024-11-03"}, "all 25 UTC hours fall on the same local date"

    def test_duplicate_local_hour_is_not_a_gap(self):
        """Gap detection runs on UTC, so a repeated local hour cannot look like missing data."""
        start = datetime(2024, 11, 3, 0, tzinfo=UTC)
        end = datetime(2024, 11, 3, 23, tzinfo=UTC)
        assert find_gaps(hour_grid(start, end), start, end) == []


class TestLocalDateBoundary:
    def test_utc_date_differs_from_local_date(self):
        """03:00 UTC on July 4th is still July 3rd in LA - the holiday off-by-one trap."""
        moment = datetime(2024, 7, 4, 3, tzinfo=UTC)
        assert local_date(moment, LA) == "2024-07-03"
        assert local_date(moment, NY) == "2024-07-03"
        assert local_date(datetime(2024, 7, 4, 12, tzinfo=UTC), LA) == "2024-07-04"

    def test_weekday_is_monday_zero(self):
        monday = datetime(2024, 1, 1, 18, tzinfo=UTC)
        feats = local_calendar_features(monday, LA)
        assert feats["dow"] == 0
        assert feats["is_weekend"] == 0

    def test_weekend_flag(self):
        saturday = datetime(2024, 1, 6, 18, tzinfo=UTC)
        assert local_calendar_features(saturday, LA)["is_weekend"] == 1


class TestGapDetection:
    def test_missing_hours_are_reported_not_filled(self):
        start = datetime(2024, 6, 1, 0, tzinfo=UTC)
        end = datetime(2024, 6, 1, 5, tzinfo=UTC)
        present = [start + timedelta(hours=i) for i in (0, 1, 4, 5)]
        assert find_gaps(present, start, end) == [
            datetime(2024, 6, 1, 2, tzinfo=UTC),
            datetime(2024, 6, 1, 3, tzinfo=UTC),
        ]

    def test_no_gaps_when_complete(self):
        start = datetime(2024, 6, 1, 0, tzinfo=UTC)
        end = datetime(2024, 6, 1, 5, tzinfo=UTC)
        assert find_gaps(hour_grid(start, end), start, end) == []

    def test_sub_hour_timestamps_are_normalized(self):
        """A value stamped 02:37 still satisfies the 02:00 slot rather than reading as a gap."""
        start = datetime(2024, 6, 1, 0, tzinfo=UTC)
        end = datetime(2024, 6, 1, 2, tzinfo=UTC)
        present = [start, start + timedelta(hours=1), datetime(2024, 6, 1, 2, 37, tzinfo=UTC)]
        assert find_gaps(present, start, end) == []

    def test_reversed_window_is_an_error(self):
        with pytest.raises(ValueError, match="precedes start"):
            hour_grid(datetime(2024, 6, 2, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))
