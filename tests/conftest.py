"""Shared fixtures. All tests run on synthetic data — no network."""

import numpy as np
import pandas as pd
import pytest

import config

# The latitude the synthetic frame "lives at". Its seasonal target cycle
# (sine peaking near the June solstice) matches a northern mid-latitude, so
# the clearness index (GHI / TOA(lat)) sees a physically consistent site.
SYNTHETIC_LAT = 45.0


@pytest.fixture
def synthetic_lat() -> float:
    return SYNTHETIC_LAT


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """A small synthetic frame shaped like fetch_history() output:
    DatetimeIndex, columns = config.NASA_POWER_PARAMETERS, no NaNs.

    The target structure is deliberately learnable (seasonal clear-sky sine
    scaled by a cloudiness multiplier) — the GBM coverage test depends on it.
    Columns are derived from config so this fixture fails loudly if the
    parameter list ever changes, instead of tests passing on a stale schema.
    """
    rng = np.random.default_rng(42)
    # ~6.5 years: LGBM_PARAMS are sized for real fetches (~10y of days);
    # on a few hundred rows the quantile models memorize the training set
    # and the coverage test below reads artificially low.
    n = 2400
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    doy = idx.dayofyear.to_numpy()

    clrsky = 4.0 + 3.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    clrsky = np.clip(clrsky, 0.5, None)
    cloudiness = rng.uniform(0.3, 1.0, n)

    special = {
        "ALLSKY_SFC_SW_DWN": clrsky * cloudiness,
        "T2M": 10 + 8 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2, n),
    }
    df = pd.DataFrame(
        {
            p: special.get(p, rng.uniform(0.5, 50, n))
            for p in config.NASA_POWER_PARAMETERS
        },
        index=idx,
    )
    assert set(df.columns) == set(config.NASA_POWER_PARAMETERS)
    return df
