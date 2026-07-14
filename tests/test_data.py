"""Feature engineering, split, and cache-staleness contracts (no network)."""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import requests

import config
from src import data
from src.data import _build_features, chronological_split, engineer_features


def test_features_complete_and_nan_free(synthetic_raw, synthetic_lat):
    out = engineer_features(synthetic_raw, synthetic_lat)
    missing = set(config.FEATURES) - set(out.columns)
    assert not missing, f"engineer_features did not build: {missing}"
    assert not out[config.FEATURES].isna().any().any()
    assert not out[config.TARGET_COL].isna().any()


@pytest.mark.parametrize("col", config.NASA_POWER_PARAMETERS)
def test_features_no_leakage(synthetic_raw, synthetic_lat, col):
    """Perturbing ANY raw input at day t must leave features for all days
    <= t unchanged — predicting day t may only use information through the
    end of day t-1. Checking every row up to t (not just t itself) also
    catches backward flow of future information, e.g. a centered rolling
    window. Parametrized over every raw column so a failure names the
    leaking input; an un-lagged weather covariate would otherwise be
    invisible to a target-only check."""
    baseline = engineer_features(synthetic_raw, synthetic_lat)
    t = baseline.index[len(baseline) // 2]  # well past the warm-up window

    perturbed_raw = synthetic_raw.copy()
    perturbed_raw.loc[t, col] += 5.0
    perturbed = engineer_features(perturbed_raw, synthetic_lat)

    pd.testing.assert_frame_equal(
        baseline.loc[:t, config.FEATURES], perturbed.loc[:t, config.FEATURES]
    )


def test_chronological_split_order_and_fractions(synthetic_raw, synthetic_lat):
    df = engineer_features(synthetic_raw, synthetic_lat)
    train, val, test = chronological_split(df)

    # nothing lost or duplicated, original order preserved
    recombined = pd.concat([train, val, test])
    pd.testing.assert_frame_equal(recombined, df)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

    f_train, f_val, f_test = config.SPLIT_FRACTIONS
    n = len(df)
    assert abs(len(train) / n - f_train) < 0.01
    assert abs(len(val) / n - f_val) < 0.01
    assert abs(len(test) / n - f_test) < 0.01


def test_lag_features_respect_calendar_gaps(synthetic_raw, synthetic_lat):
    """Delete one interior calendar day entirely (the row, not just its
    values — mimicking what the parser used to produce). Because shift() is
    positional, an un-reindexed frame would make every lag on the day after
    the hole silently read the value from one day too far back. The
    contract: those lags are NaN (the day is genuinely unobserved), rolling
    windows are calendar-true (they skip the hole, never span an extra
    day), and rows whose feature vector is thereby incomplete fall out of
    the public output."""
    target = config.TARGET_COL
    hole = synthetic_raw.index[45]  # interior, past the rolling warm-up
    day_after = hole + pd.Timedelta(days=1)
    gappy = synthetic_raw.drop(index=hole)

    built = _build_features(gappy, synthetic_lat)

    # lag1 on the day after the hole is unknown — NOT the value from two
    # days prior (which is what the pre-fix positional shift produced)
    assert np.isnan(built.loc[day_after, f"{target}_lag1"])

    # rolling window ending at the hole's edge: mean over the days that
    # exist inside the 3-day calendar window, not 3 positional rows
    # (which would smuggle in a 4th calendar day)
    y = synthetic_raw[target]
    calendar_true = y[[hole - pd.Timedelta(days=2), hole - pd.Timedelta(days=1)]].mean()
    positional_wrong = y[hole - pd.Timedelta(days=3) : hole - pd.Timedelta(days=1)].mean()
    got = built.loc[day_after, f"{target}_roll3_mean"]
    assert got == pytest.approx(calendar_true)
    assert got != pytest.approx(positional_wrong)

    # public contract: the hole itself (no observed target) and the day
    # after it (unknown lag1) must not survive into the engineered output
    out = engineer_features(gappy, synthetic_lat)
    assert hole not in out.index
    assert day_after not in out.index


def test_engineer_features_sorts_unsorted_input(synthetic_raw, synthetic_lat):
    """Contract: engineer_features sorts its input defensively (rather than
    raising on unsorted frames) — shift/rolling features are only correct on
    chronological order, and sorting makes the function safe for any caller
    (dashboard, notebooks, cached reads). A shuffled frame must therefore
    produce exactly the same output as the sorted original."""
    baseline = engineer_features(synthetic_raw, synthetic_lat)
    shuffled = engineer_features(
        synthetic_raw.sample(frac=1, random_state=0), synthetic_lat
    )

    assert shuffled.index.is_monotonic_increasing
    # check_freq=False: sample() strips the DatetimeIndex freq metadata,
    # which is irrelevant to the contract
    pd.testing.assert_frame_equal(shuffled, baseline, check_freq=False)


# ---------------------------------------------------------------------------
# Cache staleness (fetch_history) — network and clock are monkeypatched seams
# ---------------------------------------------------------------------------
TODAY = date(2026, 7, 14)
NRT_LAG_DAYS = 6  # a fetch made today observes data up to ~today - 6


def _frame_ending(last_day: date, days: int = 40) -> pd.DataFrame:
    """A cache-shaped frame (columns from config, per testing conventions)
    whose last observed (non-NaN target) day is `last_day`."""
    idx = pd.date_range(end=pd.Timestamp(last_day), periods=days, freq="D")
    return pd.DataFrame(
        {p: np.full(days, 1.0) for p in config.NASA_POWER_PARAMETERS}, index=idx
    )


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Isolated cache dir + frozen clock + recording fake network fetch.
    Returns the recorder; `.calls` counts network hits, `.frame` is what a
    successful fetch returns (fresh data: last observed = today - NRT lag)."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_today", lambda: TODAY)

    class Recorder:
        def __init__(self):
            self.calls = 0
            self.frame = _frame_ending(TODAY - timedelta(days=NRT_LAG_DAYS))
            self.error: Exception | None = None

        def __call__(self, lat, lon):
            self.calls += 1
            if self.error is not None:
                raise self.error
            return self.frame

    recorder = Recorder()
    monkeypatch.setattr(data, "_fetch_remote", recorder)
    return recorder


def _seed_cache(last_day: date) -> pd.DataFrame:
    path = data._cache_path(50.0, 0.0)
    _frame_ending(last_day).to_parquet(path)
    # return the parquet round-trip (index dtype narrows to datetime64[ms],
    # freq metadata is dropped) so comparisons match what fetch_history reads
    return pd.read_parquet(path)


def test_stale_cache_triggers_full_refetch_and_rewrite(cache_env):
    _seed_cache(TODAY - timedelta(days=20))

    out = data.fetch_history(50.0, 0.0)

    assert cache_env.calls == 1
    pd.testing.assert_frame_equal(out, cache_env.frame)
    # parquet was overwritten, not just returned fresh in-memory
    on_disk = pd.read_parquet(data._cache_path(50.0, 0.0))
    assert on_disk.index.max() == cache_env.frame.index.max()


def test_fresh_cache_never_triggers_refetch_loop(cache_env):
    """The no-refetch-loop property: a cache fetched THIS morning has a last
    observed day of ~today - NRT lag (~6 days). That must read as fresh —
    if it didn't, every request would refetch forever, since no fetch can
    ever return data newer than the NRT lag allows."""
    cached = _seed_cache(TODAY - timedelta(days=NRT_LAG_DAYS))

    out = data.fetch_history(50.0, 0.0)

    assert cache_env.calls == 0
    pd.testing.assert_frame_equal(out, cached)


def test_force_refresh_refetches_even_when_fresh(cache_env):
    _seed_cache(TODAY - timedelta(days=NRT_LAG_DAYS))

    out = data.fetch_history(50.0, 0.0, force_refresh=True)

    assert cache_env.calls == 1
    pd.testing.assert_frame_equal(out, cache_env.frame)


def test_failed_refetch_serves_stale_cache_with_warning(cache_env, caplog):
    """A live demo degrades to slightly-old data, never a stack trace."""
    cached = _seed_cache(TODAY - timedelta(days=20))
    cache_env.error = requests.ConnectionError("NASA POWER is down")

    with caplog.at_level(logging.WARNING, logger="src.data"):
        out = data.fetch_history(50.0, 0.0)

    assert cache_env.calls == 1
    pd.testing.assert_frame_equal(out, cached)
    assert any("stale cache" in r.message for r in caplog.records)


def test_cache_exactly_at_threshold_is_not_stale(cache_env):
    """Boundary pin: staleness is strictly '>', so a last observed day of
    exactly today - CACHE_MAX_AGE_DAYS does not refetch."""
    cached = _seed_cache(TODAY - timedelta(days=config.CACHE_MAX_AGE_DAYS))

    out = data.fetch_history(50.0, 0.0)

    assert cache_env.calls == 0
    pd.testing.assert_frame_equal(out, cached)
