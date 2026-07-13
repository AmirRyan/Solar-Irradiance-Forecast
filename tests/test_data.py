"""Feature engineering + split contracts (no network)."""

import numpy as np
import pandas as pd
import pytest

import config
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
