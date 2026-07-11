"""Pure dashboard-layer helpers (no dash import, no server, no network)."""

import config
from dashboard.helpers import ResultMemo, memo_key, resolve_location, validate_latlon
from src.pipeline import MIN_FIT_DAYS, MIN_TEST_DAYS, check_sufficient_history

import pytest


# --- resolve_location: the input that fired wins --------------------------
def test_trigger_wins_over_stale_state():
    """Regression test for the scaffold bug: preset selection must win even
    when previously typed lat/lon values are still sitting in their fields."""
    preset = next(iter(config.DEMO_LOCATIONS))
    expected = config.DEMO_LOCATIONS[preset]
    loc = resolve_location("preset", preset, None, lat=60.39, lon=5.32)
    assert loc == (expected["lat"], expected["lon"])


def test_go_uses_manual_fields():
    assert resolve_location("go", "Bristol, UK", None, 10.5, -20.25) == (10.5, -20.25)


def test_map_click_resolves_point():
    click = {"points": [{"lat": -1.29, "lon": 36.82}]}
    assert resolve_location("map", None, click, None, None) == (-1.29, 36.82)


def test_cleared_preset_and_empty_click_resolve_to_none():
    assert resolve_location("preset", None, None, 51.0, -2.0) is None
    assert resolve_location("preset", "Atlantis", None, None, None) is None
    assert resolve_location("map", None, {"points": []}, None, None) is None
    assert resolve_location(None, "Bristol, UK", None, None, None) is None


# --- validate_latlon -------------------------------------------------------
def test_validate_latlon_accepts_valid():
    assert validate_latlon(51.4545, -2.5879) == (51.4545, -2.5879)
    assert validate_latlon(-90, 180) == (-90.0, 180.0)


@pytest.mark.parametrize(
    "lat,lon",
    [(91, 0), (-90.01, 0), (0, 181), (0, -180.5), (None, 0), (0, None), ("abc", 0)],
)
def test_validate_latlon_rejects_invalid(lat, lon):
    result = validate_latlon(lat, lon)
    assert isinstance(result, str) and result  # readable message, not a tuple


# --- memo ------------------------------------------------------------------
def test_memo_key_matches_data_cache_rounding():
    """The memo and data.py's parquet cache must agree on location identity
    (both round to 2dp)."""
    from src.data import _cache_path

    lat, lon = 51.4545, -2.5879
    assert memo_key(lat, lon) == (51.45, -2.59)
    assert f"{memo_key(lat, lon)[0]:.2f}_{memo_key(lat, lon)[1]:.2f}" in _cache_path(lat, lon).name


def test_memo_lru_eviction_and_hit():
    memo = ResultMemo(maxsize=2)
    memo.put(1.0, 1.0, "a")
    memo.put(2.0, 2.0, "b")
    assert memo.get(1.0, 1.0) == "a"          # touch: 1.0 becomes most recent
    assert memo.get(1.001, 1.004) == "a"      # same 2dp key -> hit
    memo.put(3.0, 3.0, "c")                   # evicts 2.0 (least recent)
    assert memo.get(2.0, 2.0) is None
    assert memo.get(1.0, 1.0) == "a"
    assert len(memo) == 2


# --- pipeline history guard --------------------------------------------------
def test_history_guard_passes_on_normal_history():
    check_sufficient_history(2800, 600)  # ~10y of NASA POWER data


@pytest.mark.parametrize(
    "n_fit,n_test", [(MIN_FIT_DAYS - 1, 100), (500, MIN_TEST_DAYS - 1), (0, 0)]
)
def test_history_guard_raises_readably(n_fit, n_test):
    with pytest.raises(ValueError, match="usable days of history"):
        check_sufficient_history(n_fit, n_test)
