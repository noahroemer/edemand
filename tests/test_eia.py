"""EIA client tests, against mocked HTTP.

The behaviors worth pinning are the ones that would silently corrupt data rather than
raise: pagination arithmetic, string-to-float coercion, missing-vs-zero, and the UTC
frequency choice.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import responses

from ingest import eia

START = datetime(2024, 6, 1, 0, tzinfo=UTC)
END = datetime(2024, 6, 1, 5, tzinfo=UTC)


def _page(rows: list[dict], total: int) -> dict:
    return {"response": {"total": str(total), "data": rows}}


def _row(hour: int, value: object = "25000.5") -> dict:
    return {"period": f"2024-06-01T{hour:02d}", "respondent": "CISO", "value": value}


@responses.activate
def test_single_page_fetch():
    responses.add(
        responses.GET,
        eia.BASE_URL,
        json=_page([_row(h) for h in range(6)], total=6),
        status=200,
    )
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert len(rows) == 6
    assert rows[0]["period_utc"] == datetime(2024, 6, 1, 0, tzinfo=UTC)
    assert rows[0]["demand_mw"] == 25000.5
    assert rows[0]["region"] == "CISO"
    assert rows[0]["data_type"] == "D"


@responses.activate
def test_requests_utc_frequency_not_local():
    """CLAUDE.md section 4 requires UTC storage; 'local-hourly' would break it."""
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0)], total=1), status=200)
    eia.fetch_demand("CISO", START, END, api_key="fake")
    query = responses.calls[0].request.params
    assert query["frequency"] == "hourly"
    assert query["facets[type][]"] == "D"
    assert query["facets[respondent][]"] == "CISO"


@responses.activate
def test_requests_stable_sort_for_pagination():
    """Offset pagination without an explicit sort can duplicate or skip rows."""
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0)], total=1), status=200)
    eia.fetch_demand("CISO", START, END, api_key="fake")
    query = responses.calls[0].request.params
    assert query["sort[0][column]"] == "period"
    assert query["sort[0][direction]"] == "asc"


@responses.activate
def test_paginates_until_total_reached():
    responses.add(
        responses.GET, eia.BASE_URL, json=_page([_row(h) for h in range(3)], 5), status=200
    )
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(h) for h in (3, 4)], 5), status=200)
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert len(rows) == 5
    assert len(responses.calls) == 2
    assert responses.calls[1].request.params["offset"] == "3"


@responses.activate
def test_stops_on_short_page_even_if_total_overstates():
    """A short page means the server is out of rows, whatever `total` claimed."""
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0)], total=9999), status=200)
    responses.add(responses.GET, eia.BASE_URL, json=_page([], total=9999), status=200)
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert len(rows) == 1


@responses.activate
def test_duplicate_periods_across_pages_are_deduplicated():
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0), _row(1)], 4), status=200)
    responses.add(
        responses.GET,
        eia.BASE_URL,
        json=_page([_row(1, "30000"), _row(2)], 4),
        status=200,
    )
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    periods = [r["period_utc"] for r in rows]
    assert len(periods) == len(set(periods))
    # Last occurrence wins, so behavior is deterministic rather than order-dependent.
    hour_one = next(r for r in rows if r["period_utc"].hour == 1)
    assert hour_one["demand_mw"] == 30000.0


@responses.activate
def test_rows_are_sorted_by_period():
    responses.add(
        responses.GET,
        eia.BASE_URL,
        json=_page([_row(5), _row(0), _row(3)], total=3),
        status=200,
    )
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert [r["period_utc"].hour for r in rows] == [0, 3, 5]


@responses.activate
@pytest.mark.parametrize("missing", [None, ""])
def test_missing_value_becomes_none_not_zero(missing):
    """Zero demand is physically meaningful; conflating it with missing corrupts the gap flag."""
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0, missing)], total=1), status=200)
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert rows[0]["demand_mw"] is None


@responses.activate
def test_genuine_zero_is_preserved():
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0, "0")], total=1), status=200)
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert rows[0]["demand_mw"] == 0.0


@responses.activate
def test_non_numeric_value_is_treated_as_missing():
    responses.add(responses.GET, eia.BASE_URL, json=_page([_row(0, "n/a")], total=1), status=200)
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert rows[0]["demand_mw"] is None


@responses.activate
def test_403_names_the_key_as_the_problem():
    responses.add(responses.GET, eia.BASE_URL, json={}, status=403)
    with pytest.raises(eia.EIAError, match="403"):
        eia.fetch_demand("CISO", START, END, api_key="bad")


@responses.activate
def test_malformed_envelope_raises():
    responses.add(responses.GET, eia.BASE_URL, json={"warnings": ["no data"]}, status=200)
    with pytest.raises(eia.EIAError, match="Unexpected envelope"):
        eia.fetch_demand("CISO", START, END, api_key="fake")


def test_missing_key_points_at_registration(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    with pytest.raises(eia.EIAError, match="register"):
        eia.resolve_api_key(None)


def test_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("EIA_API_KEY", "from-env")
    assert eia.resolve_api_key(None) == "from-env"
    assert eia.resolve_api_key("explicit") == "explicit"


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="Naive datetime"):
        eia.fetch_demand("CISO", datetime(2024, 6, 1), END, api_key="fake")  # noqa: DTZ001


@responses.activate
def test_fetched_at_is_shared_across_the_call():
    """One revision stamp per fetch - per-row stamps would imply separate observations."""
    responses.add(
        responses.GET, eia.BASE_URL, json=_page([_row(h) for h in range(3)], 3), status=200
    )
    rows = eia.fetch_demand("CISO", START, END, api_key="fake")
    assert len({r["fetched_at"] for r in rows}) == 1
