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


# ---------------------------------------------------------------------------
# Subregion route. The CAISO sub-areas are only available here - `region-data`
# returns an empty result for them rather than an error, which would look like
# "no data exists" and quietly break the hierarchical model.
# ---------------------------------------------------------------------------


def _sub_row(hour: int, subba: str, value: object = "12000") -> dict:
    return {
        "period": f"2024-06-01T{hour:02d}",
        "parent": "CISO",
        "subba": subba,
        "value": value,
    }


@responses.activate
def test_subba_uses_the_subregion_route_and_facets():
    responses.add(
        responses.GET, eia.SUBBA_URL, json=_page([_sub_row(0, "PGAE")], total=1), status=200
    )
    eia.fetch_subba_demand("CISO", START, END, api_key="fake")
    request = responses.calls[0].request
    assert eia.SUBBA_URL in request.url
    assert request.params["facets[parent][]"] == "CISO"
    assert "facets[respondent][]" not in request.params
    # This route publishes demand only - there is no type facet to send.
    assert "facets[type][]" not in request.params


@responses.activate
def test_subba_deduplicates_on_subba_and_period_not_period_alone():
    """One period carries one row per subregion. Deduplicating on period would throw
    away three of every four rows."""
    rows = [_sub_row(0, s) for s in ("PGAE", "SCE", "SDGE", "VEA")]
    responses.add(responses.GET, eia.SUBBA_URL, json=_page(rows, total=4), status=200)
    out = eia.fetch_subba_demand("CISO", START, END, api_key="fake")
    assert len(out) == 4
    assert {r["region"] for r in out} == {"PGAE", "SCE", "SDGE", "VEA"}
    assert {r["period_utc"] for r in out} == {datetime(2024, 6, 1, 0, tzinfo=UTC)}


@responses.activate
def test_subba_tags_parent_for_the_mint_summation_matrix():
    responses.add(
        responses.GET, eia.SUBBA_URL, json=_page([_sub_row(0, "SCE")], total=1), status=200
    )
    out = eia.fetch_subba_demand("CISO", START, END, api_key="fake")
    assert out[0]["parent_region"] == "CISO"
    assert out[0]["region"] == "SCE"


@responses.activate
def test_subba_filter_narrows_to_requested_subregions():
    responses.add(
        responses.GET, eia.SUBBA_URL, json=_page([_sub_row(0, "SCE")], total=1), status=200
    )
    eia.fetch_subba_demand("CISO", START, END, subbas=["SCE", "SDGE"], api_key="fake")
    params = responses.calls[0].request.params
    assert params["facets[subba][0]"] == "SCE"
    assert params["facets[subba][1]"] == "SDGE"


@responses.activate
def test_subba_sorted_by_period_then_subregion():
    rows = [_sub_row(1, "SCE"), _sub_row(0, "VEA"), _sub_row(0, "PGAE")]
    responses.add(responses.GET, eia.SUBBA_URL, json=_page(rows, total=3), status=200)
    out = eia.fetch_subba_demand("CISO", START, END, api_key="fake")
    assert [(r["period_utc"].hour, r["region"]) for r in out] == [
        (0, "PGAE"),
        (0, "VEA"),
        (1, "SCE"),
    ]


@pytest.mark.network
def test_live_subba_returns_all_four_caiso_subareas(eia_key):
    rows = eia.fetch_subba_demand(
        "CISO", datetime(2024, 7, 1, 0, tzinfo=UTC), datetime(2024, 7, 1, 23, tzinfo=UTC)
    )
    assert len(rows) == 96, "expected 24 hours x 4 subregions"
    assert {r["region"] for r in rows} == {"PGAE", "SCE", "SDGE", "VEA"}


@pytest.mark.network
def test_live_hierarchy_is_coherent(eia_key):
    """The sum of subregions should equal the parent total. MinT reconciliation assumes
    this structure; if EIA's published values disagreed materially, the coherence
    constraint would be imposing a relationship the data does not have."""
    window = (datetime(2024, 7, 1, 0, tzinfo=UTC), datetime(2024, 7, 1, 23, tzinfo=UTC))
    subs = eia.fetch_subba_demand("CISO", *window)
    totals = {r["period_utc"]: r["demand_mw"] for r in eia.fetch_demand("CISO", *window)}

    summed: dict[datetime, float] = {}
    for row in subs:
        if row["demand_mw"] is not None:
            summed[row["period_utc"]] = summed.get(row["period_utc"], 0.0) + row["demand_mw"]

    worst = max(abs(summed[p] - totals[p]) / totals[p] for p in summed if totals.get(p))
    assert worst < 0.01, f"subregions disagree with parent by {worst:.2%}"
