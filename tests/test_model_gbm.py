"""QuantileGBM prediction contract (no network)."""

import config
from src.data import chronological_split, engineer_features
from src.model_gbm import QuantileGBM


def test_quantiles_are_monotone_and_calibrated(synthetic_raw, synthetic_lat):
    df = engineer_features(synthetic_raw, synthetic_lat)
    train, val, test = chronological_split(df)

    gbm = QuantileGBM().fit(train)
    preds = gbm.predict(test)

    # Contract guard: monotonicity is enforced by the per-row sort in
    # predict(), but it is part of the public contract, so pin it.
    assert list(preds.columns) == ["p10", "p50", "p90"]
    assert preds.index.equals(test.index)
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()
    assert not preds.isna().any().any()

    # Statistical check: ideally the P10-P90 band covers ~80% of actuals.
    # With the phase-2 tuned LGBM_PARAMS this fixture measures 0.733 and the
    # real demo locations 0.72-0.78, so the lower bound is 0.65 — loose
    # enough not to flake, tight enough to catch a return of the
    # undercoverage regression this tuning pass fixed.
    y = test[config.TARGET_COL]
    coverage = ((y >= preds["p10"]) & (y <= preds["p90"])).mean()
    assert 0.65 <= coverage <= 0.95, f"P10-P90 coverage {coverage:.2f} outside [0.65, 0.95]"
