"""Open-Meteo client tests.

Mostly mocked. The `network`-marked tests at the bottom hit the real API — Open-Meteo needs
no key, so they are genuinely runnable, and they are the only end-to-end proof the request
shape is right. Deselect with `-m "not network"`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import responses

from ingest import open_meteo as om

LA = (34.05, -118.24)
NYC = (40.71, -74.01)


def _block(times: list[str], temps: list[float], humidity: list[float] | None = None) -> dict:
    return {
        "latitude": 34.05,
        "longitude": -118.24,
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": humidity if humidity is not None else [50.0] * len(times),
        },
    }


@responses.activate
def test_parses_hourly_block_into_rows():
    responses.add(
        responses.GET,
        om.ARCHIVE_URL,
        json=_block(["2024-06-01T00:00", "2024-06-01T01:00"], [18.5, 17.9]),
        status=200,
    )
    rows = om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    assert len(rows) == 2
    assert rows[0]["period_utc"] == datetime(2024, 6, 1, 0, tzinfo=UTC)
    assert rows[0]["temp_c"] == 18.5
    assert rows[0]["humidity_pct"] == 50.0
    assert rows[0]["source"] == "era5_archive"


@responses.activate
def test_requests_gmt_explicitly():
    """Open-Meteo defaults to GMT, but section 4's UTC rule is too important to leave implicit."""
    responses.add(
        responses.GET, om.ARCHIVE_URL, json=_block(["2024-06-01T00:00"], [18.0]), status=200
    )
    om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    assert responses.calls[0].request.params["timezone"] == "GMT"


@responses.activate
def test_timestamps_are_tagged_utc_not_naive():
    responses.add(
        responses.GET, om.ARCHIVE_URL, json=_block(["2024-06-01T00:00"], [18.0]), status=200
    )
    rows = om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    assert rows[0]["period_utc"].tzinfo is not None
    assert rows[0]["period_utc"].utcoffset().total_seconds() == 0


@responses.activate
def test_batches_multiple_coordinates_in_one_request():
    responses.add(
        responses.GET,
        om.ARCHIVE_URL,
        json=[
            _block(["2024-06-01T00:00"], [18.0]),
            _block(["2024-06-01T00:00"], [24.0]),
        ],
        status=200,
    )
    rows = om.fetch_archive([LA, NYC], "2024-06-01", "2024-06-01")
    assert len(responses.calls) == 1
    assert len(rows) == 2
    # Response order matches request order, which is what lets results map back to regions.
    assert rows[0]["latitude"] == LA[0]
    assert rows[1]["latitude"] == NYC[0]
    assert rows[0]["temp_c"] == 18.0
    assert rows[1]["temp_c"] == 24.0


@responses.activate
def test_block_count_mismatch_raises():
    """Silently zipping mismatched lists would attach the wrong weather to a region."""
    responses.add(
        responses.GET, om.ARCHIVE_URL, json=[_block(["2024-06-01T00:00"], [18.0])], status=200
    )
    with pytest.raises(om.OpenMeteoError, match="got 1 blocks"):
        om.fetch_archive([LA, NYC], "2024-06-01", "2024-06-01")


@responses.activate
def test_null_values_are_preserved_as_none():
    responses.add(
        responses.GET,
        om.ARCHIVE_URL,
        json=_block(["2024-06-01T00:00", "2024-06-01T01:00"], [18.0, None]),
        status=200,
    )
    rows = om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    assert rows[1]["temp_c"] is None


@responses.activate
def test_error_body_is_surfaced():
    responses.add(
        responses.GET,
        om.ARCHIVE_URL,
        json={"error": True, "reason": "start_date is after end_date"},
        status=400,
    )
    with pytest.raises(om.OpenMeteoError, match="400"):
        om.fetch_archive([LA], "2024-06-02", "2024-06-01")


@responses.activate
def test_missing_hourly_block_raises():
    responses.add(responses.GET, om.ARCHIVE_URL, json={"latitude": 34.05}, status=200)
    with pytest.raises(om.OpenMeteoError, match="no hourly.time"):
        om.fetch_archive([LA], "2024-06-01", "2024-06-01")


def test_empty_coordinates_rejected():
    with pytest.raises(ValueError, match="No coordinates"):
        om.fetch_archive([], "2024-06-01", "2024-06-01")


@responses.activate
def test_each_endpoint_tags_its_own_source():
    """The source column is how a backtest proves which weather it used - see section 7."""
    for url, fn, expected in (
        (om.ARCHIVE_URL, om.fetch_archive, "era5_archive"),
        (om.HISTORICAL_FORECAST_URL, om.fetch_historical_forecast, "historical_forecast"),
    ):
        responses.reset()
        responses.add(responses.GET, url, json=_block(["2024-06-01T00:00"], [18.0]), status=200)
        rows = fn([LA], "2024-06-01", "2024-06-01")
        assert rows[0]["source"] == expected


@responses.activate
def test_previous_day_maps_to_standard_columns_and_tags_source():
    """Previous Runs variables arrive with a _previous_day1 suffix but must land in the same
    temp_c / humidity_pct columns, so silver and gold treat every source identically."""
    responses.add(
        responses.GET,
        om.PREVIOUS_RUNS_URL,
        json={
            "latitude": 34.05,
            "longitude": -118.24,
            "hourly": {
                "time": ["2025-01-20T06:00"],
                "temperature_2m_previous_day1": [-5.0],
                "relative_humidity_2m_previous_day1": [61],
            },
        },
        status=200,
    )
    rows = om.fetch_previous_day([LA], "2025-01-20", "2025-01-20")
    assert rows[0]["temp_c"] == -5.0
    assert rows[0]["humidity_pct"] == 61
    assert rows[0]["source"] == "previous_day1"
    assert "temperature_2m_previous_day1" in responses.calls[0].request.url


@responses.activate
def test_forecast_requests_two_days_by_default():
    """The 24h horizon issued at 06:00 local runs into the next calendar day."""
    responses.add(
        responses.GET, om.FORECAST_URL, json=_block(["2024-06-01T00:00"], [18.0]), status=200
    )
    om.fetch_forecast([LA])
    assert responses.calls[0].request.params["forecast_days"] == "2"


# ---------------------------------------------------------------------------
# Live API tests. No key needed, so these actually run: pytest -m network
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_archive_returns_a_full_day():
    rows = om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    assert len(rows) == 24
    assert all(r["period_utc"].tzinfo is not None for r in rows)
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    assert temps, "no temperature values returned"
    assert all(-60.0 < t < 60.0 for t in temps), f"implausible temperatures: {temps}"


@pytest.mark.network
def test_live_batched_request_maps_back_in_order():
    rows = om.fetch_archive([LA, NYC], "2024-06-01", "2024-06-01")
    assert len(rows) == 48
    assert {r["latitude"] for r in rows} == {LA[0], NYC[0]}


@pytest.mark.network
def test_live_historical_forecast_differs_from_reanalysis():
    """The point of section 7: archived forecasts are not reanalysis. If these were
    identical there would be no leakage concern to disclose."""
    archive = om.fetch_archive([LA], "2024-06-01", "2024-06-01")
    forecast = om.fetch_historical_forecast([LA], "2024-06-01", "2024-06-01")
    assert len(forecast) == 24
    pairs = [
        (a["temp_c"], f["temp_c"])
        for a, f in zip(archive, forecast, strict=True)
        if a["temp_c"] is not None and f["temp_c"] is not None
    ]
    assert pairs, "no comparable hours"
    assert any(a != f for a, f in pairs), "archive and archived-forecast were identical"


@pytest.mark.network
def test_live_forecast_covers_the_horizon():
    rows = om.fetch_forecast([LA], forecast_days=2)
    assert len(rows) >= 24
