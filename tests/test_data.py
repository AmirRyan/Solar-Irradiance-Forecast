"""Feature engineering + split contracts (no network)."""

import pandas as pd
import pytest

import config
from src.data import chronological_split, engineer_features


def test_features_complete_and_nan_free(synthetic_raw):
    out = engineer_features(synthetic_raw)
    missing = set(config.FEATURES) - set(out.columns)
    assert not missing, f"engineer_features did not build: {missing}"
    assert not out[config.FEATURES].isna().any().any()
    assert not out[config.TARGET_COL].isna().any()


@pytest.mark.parametrize("col", config.NASA_POWER_PARAMETERS)
def test_features_no_leakage(synthetic_raw, col):
    """Perturbing ANY raw input at day t must leave features for all days
    <= t unchanged — predicting day t may only use information through the
    end of day t-1. Checking every row up to t (not just t itself) also
    catches backward flow of future information, e.g. a centered rolling
    window. Parametrized over every raw column so a failure names the
    leaking input; an un-lagged weather covariate would otherwise be
    invisible to a target-only check."""
    baseline = engineer_features(synthetic_raw)
    t = baseline.index[len(baseline) // 2]  # well past the warm-up window

    perturbed_raw = synthetic_raw.copy()
    perturbed_raw.loc[t, col] += 5.0
    perturbed = engineer_features(perturbed_raw)

    pd.testing.assert_frame_equal(
        baseline.loc[:t, config.FEATURES], perturbed.loc[:t, config.FEATURES]
    )


def test_chronological_split_order_and_fractions(synthetic_raw):
    df = engineer_features(synthetic_raw)
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


def test_engineer_features_sorts_unsorted_input(synthetic_raw):
    """Contract: engineer_features sorts its input defensively (rather than
    raising on unsorted frames) — shift/rolling features are only correct on
    chronological order, and sorting makes the function safe for any caller
    (dashboard, notebooks, cached reads). A shuffled frame must therefore
    produce exactly the same output as the sorted original."""
    baseline = engineer_features(synthetic_raw)
    shuffled = engineer_features(synthetic_raw.sample(frac=1, random_state=0))

    assert shuffled.index.is_monotonic_increasing
    # check_freq=False: sample() strips the DatetimeIndex freq metadata,
    # which is irrelevant to the contract
    pd.testing.assert_frame_equal(shuffled, baseline, check_freq=False)
