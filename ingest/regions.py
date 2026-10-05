"""Region metadata: EIA balancing-authority codes, timezones, weather coordinates.

The 10 regions and the 4 CAISO sub-areas come from CLAUDE.md section 4.

**On the coordinates:** each region gets ONE representative point, chosen as the largest
load center rather than the geographic centroid — demand follows people, not area. These
are approximations and they are the weakest assumption in the weather pipeline: MISO alone
spans Minnesota to Louisiana, so a single temperature reading is a crude proxy for its load
response. Upgrading to several points per region, population-weighted, is the obvious
improvement if weather features underperform. Flagged here so the choice is visible rather
than buried.

Timezones are the IANA identifier for the region's dominant local time. They exist to build
calendar features (hour_local, dow, is_holiday) from UTC timestamps, per section 4. Regions
spanning two zones (MISO, SWPP) use the one covering most of their load.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Region:
    """One balancing authority."""

    code: str
    """EIA respondent code, e.g. 'CISO'. The join key everywhere downstream."""

    name: str
    timezone: str
    """IANA timezone for local calendar features."""

    latitude: float
    longitude: float
    """Representative load center - an approximation, see module docstring."""

    load_center: str
    """Which city the coordinates point at, so the approximation is auditable."""

    holiday_subdiv: str | None = None
    """US state code for state-level holidays, or None for federal-only. Multi-state
    regions get federal-only because no single state's calendar represents them."""


# CLAUDE.md section 4: the initial region set.
REGIONS: dict[str, Region] = {
    r.code: r
    for r in (
        Region(
            "CISO",
            "California ISO",
            "America/Los_Angeles",
            34.05,
            -118.24,
            "Los Angeles basin",
            "CA",
        ),
        Region(
            "ERCO", "ERCOT (Texas)", "America/Chicago", 32.78, -96.80, "Dallas-Fort Worth", "TX"
        ),
        Region("PJM", "PJM Interconnection", "America/New_York", 39.95, -75.17, "Philadelphia"),
        Region("MISO", "Midcontinent ISO", "America/Chicago", 39.77, -86.16, "Indianapolis"),
        Region("NYIS", "New York ISO", "America/New_York", 40.71, -74.01, "New York City", "NY"),
        Region("ISNE", "ISO New England", "America/New_York", 42.36, -71.06, "Boston", "MA"),
        Region(
            "BPAT",
            "Bonneville Power Administration",
            "America/Los_Angeles",
            45.52,
            -122.68,
            "Portland",
            "OR",
        ),
        Region("SWPP", "Southwest Power Pool", "America/Chicago", 37.69, -97.34, "Wichita"),
        Region("FPL", "Florida Power & Light", "America/New_York", 25.76, -80.19, "Miami", "FL"),
        Region(
            "LDWP",
            "LA Dept of Water & Power",
            "America/Los_Angeles",
            34.05,
            -118.24,
            "Los Angeles",
            "CA",
        ),
    )
}

# CAISO sub-areas for the hierarchical model (section 4). These are children of CISO:
# their demand should sum to the CISO total, which is what MinT reconciliation enforces.
CAISO_SUBAREAS: dict[str, Region] = {
    r.code: r
    for r in (
        Region(
            "PGAE",
            "Pacific Gas & Electric",
            "America/Los_Angeles",
            37.77,
            -122.42,
            "San Francisco",
            "CA",
        ),
        Region(
            "SCE",
            "Southern California Edison",
            "America/Los_Angeles",
            34.14,
            -118.14,
            "Pasadena",
            "CA",
        ),
        Region(
            "SDGE",
            "San Diego Gas & Electric",
            "America/Los_Angeles",
            32.72,
            -117.16,
            "San Diego",
            "CA",
        ),
        Region(
            "VEA",
            "Valley Electric Association",
            "America/Los_Angeles",
            36.21,
            -115.98,
            "Pahrump",
            "NV",
        ),
    )
}

# NOTE: LDWP is a peer of CISO in REGIONS, not a CAISO child - LADWP is its own balancing
# authority and is not inside the CAISO footprint. Treating it as a child would break the
# hierarchy's coherence constraint.
CAISO_PARENT = "CISO"

ALL_REGIONS: dict[str, Region] = {**REGIONS, **CAISO_SUBAREAS}


def get(code: str) -> Region:
    """Look up a region by EIA code, case-insensitively.

    Raises KeyError with the valid options rather than returning None, so a typo fails
    loudly at ingestion time instead of silently producing an empty table.
    """
    key = code.upper()
    if key not in ALL_REGIONS:
        raise KeyError(f"Unknown region {code!r}. Known: {', '.join(sorted(ALL_REGIONS))}")
    return ALL_REGIONS[key]


def coordinates(codes: list[str] | None = None) -> list[tuple[float, float]]:
    """(lat, lon) pairs in the order given. Open-Meteo accepts batched coordinates in one
    request, so keeping order stable matters for zipping results back to regions."""
    selected = codes or list(REGIONS)
    return [(get(c).latitude, get(c).longitude) for c in selected]
