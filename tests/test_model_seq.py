"""SeqModel contract (no network). Skipped when the optional torch extra is
not installed."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import config  # noqa: E402
from src.data import chronological_split, engineer_features  # noqa: E402
from src.model_seq import SeqModel  # noqa: E402

FAST = {"max_epochs": 4, "patience": 2}


@pytest.fixture(scope="module")
def fitted():
    # module-scoped fixtures can't use the function-scoped conftest fixtures,
    # so rebuild the same frame the conftest builds.
    from tests.conftest import SYNTHETIC_LAT
    rng = np.random.default_rng(42)
    n = 2400
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    doy = idx.dayofyear.to_numpy()
    clrsky = np.clip(4.0 + 3.0 * np.sin(2 * np.pi * (doy - 80) / 365.25), 0.5, None)
    special = {
        "ALLSKY_SFC_SW_DWN": clrsky * rng.uniform(0.3, 1.0, n),
        "T2M": 10 + 8 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2, n),
    }
    raw = pd.DataFrame(
        {p: special.get(p, rng.uniform(0.5, 50, n)) for p in config.NASA_POWER_PARAMETERS},
        index=idx,
    )
    df = engineer_features(raw, SYNTHETIC_LAT)
    train, val, test = chronological_split(df)
    fit_df = pd.concat([train, val])
    return SeqModel(**FAST).fit(fit_df), fit_df, test


def test_predict_contract(fitted):
    model, _, test = fitted
    preds = model.predict(test)
    assert list(preds.columns) == ["p10", "p50", "p90"]
    assert preds.index.equals(test.index)
    # the synthetic test frame is gapless and the fit tail supplies context,
    # so every row has a full window
    assert not preds.isna().any().any()
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()


def test_deterministic_given_seed(fitted):
    model, fit_df, test = fitted
    again = SeqModel(**FAST).fit(fit_df)
    pd.testing.assert_frame_equal(model.predict(test), again.predict(test))


def test_no_backward_flow_from_future_rows(fitted):
    model, _, test = fitted
    base = model.predict(test)
    t = test.index[len(test) // 2]
    perturbed = test.copy()
    perturbed.loc[perturbed.index > t, config.FEATURES] *= 7.0
    after = model.predict(perturbed)
    pd.testing.assert_frame_equal(base.loc[:t], after.loc[:t])


def test_calendar_gap_window_is_nan_not_misaligned(fitted):
    model, _, test = fitted
    w = config.SEQ_PARAMS["window"]
    gapped = test.drop(test.index[100])  # one missing calendar day
    preds = model.predict(gapped)
    # windows ending within `w` rows after the hole span it
    hole = test.index[100]
    spans = (preds.index > hole) & (preds.index < hole + pd.Timedelta(days=w))
    assert preds[spans].isna().all().all()
    assert preds[preds.index > hole + pd.Timedelta(days=w)].notna().all().all()


def test_coverage_sanity(fitted):
    model, _, test = fitted
    preds = model.predict(test)
    y = test[config.TARGET_COL]
    coverage = ((y >= preds["p10"]) & (y <= preds["p90"])).mean()
    assert 0.4 <= coverage <= 0.99  # short training: only guards against nonsense
