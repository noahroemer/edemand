"""Region metadata and holiday calendar tests."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from ingest import calendars, regions


class TestRegions:
    def test_ten_regions_per_design_doc(self):
        assert len(regions.REGIONS) == 10
        assert set(regions.REGIONS) == {
            "CISO",
            "ERCO",
            "PJM",
            "MISO",
            "NYIS",
            "ISNE",
            "BPAT",
            "SWPP",
            "FPL",
            "LDWP",
        }

    def test_four_caiso_subareas(self):
        assert set(regions.CAISO_SUBAREAS) == {"PGAE", "SCE", "SDGE", "VEA"}

    def test_ldwp_is_not_a_caiso_child(self):
        """LADWP is its own balancing authority outside the CAISO footprint. Treating it
        as a child would break the hierarchy's coherence constraint for MinT."""
        assert "LDWP" not in regions.CAISO_SUBAREAS
        assert "LDWP" in regions.REGIONS

    def test_lookup_is_case_insensitive(self):
        assert regions.get("ciso").code == "CISO"

    def test_unknown_region_fails_loudly(self):
        """A typo must not silently produce an empty table."""
        with pytest.raises(KeyError, match="Unknown region"):
            regions.get("NOTAREGION")

    @pytest.mark.parametrize("code", sorted(regions.ALL_REGIONS))
    def test_every_region_has_a_valid_timezone(self, code):
        ZoneInfo(regions.get(code).timezone)  # raises if the identifier is bogus

    @pytest.mark.parametrize("code", sorted(regions.ALL_REGIONS))
    def test_coordinates_are_in_the_continental_us(self, code):
        region = regions.get(code)
        assert 24.0 < region.latitude < 50.0, f"{code} latitude off-continent"
        assert -125.0 < region.longitude < -66.0, f"{code} longitude off-continent"

    def test_coordinate_ordering_is_stable(self):
        """Open-Meteo returns batched results in request order, so this mapping must hold."""
        codes = ["NYIS", "CISO", "FPL"]
        coords = regions.coordinates(codes)
        assert coords == [(regions.get(c).latitude, regions.get(c).longitude) for c in codes]

    def test_multi_state_regions_use_federal_holidays_only(self):
        for code in ("PJM", "MISO", "SWPP"):
            assert regions.get(code).holiday_subdiv is None
        assert regions.get("CISO").holiday_subdiv == "CA"


class TestHolidays:
    def test_federal_holiday_is_flagged(self):
        # 2024-07-04 18:00 UTC is 11:00 local in LA, safely inside the local day.
        assert calendars.is_holiday(datetime(2024, 7, 4, 18, tzinfo=UTC), "CISO")

    def test_ordinary_day_is_not(self):
        assert not calendars.is_holiday(datetime(2024, 7, 9, 18, tzinfo=UTC), "CISO")

    def test_holiday_uses_local_date_not_utc_date(self):
        """03:00 UTC on the 4th is still the 3rd in LA. Flagging it would shift every
        western region's holiday effect a day early."""
        early = datetime(2024, 7, 4, 3, tzinfo=UTC)
        assert not calendars.is_holiday(early, "CISO")
        assert calendars.is_holiday(datetime(2024, 7, 4, 18, tzinfo=UTC), "CISO")

    def test_holiday_name_is_available_for_regime_analysis(self):
        name = calendars.holiday_name(datetime(2024, 11, 28, 18, tzinfo=UTC), "PJM")
        assert name and "Thanksgiving" in name

    def test_state_holiday_only_applies_to_single_state_regions(self):
        """Cesar Chavez Day (March 31) is a California holiday, not federal."""
        moment = datetime(2024, 4, 1, 18, tzinfo=UTC)  # Mar 31 observed -> Apr 1 in 2024
        ca = calendars.is_holiday(moment, "CISO")
        federal_only = calendars.is_holiday(moment, "PJM")
        assert ca != federal_only, "state subdivision had no effect"

    def test_observed_date_shifts_weekend_holidays(self):
        """2021-07-04 fell on a Sunday; the observed holiday was Monday the 5th."""
        dates = calendars.holiday_dates("PJM", [2021])
        observed = {d.isoformat(): n for d, n in dates.items()}
        assert any("2021-07-05" == d for d in observed), observed

    def test_calendar_is_cached(self):
        a = calendars.holiday_calendar("CISO", [2024])
        b = calendars.holiday_calendar("CISO", [2024])
        assert a is b, "repeated lookups should hit the lru_cache"
