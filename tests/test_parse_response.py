"""_parse_power_response contract: the -999 / trailing-missing handling
that fetch_history applies to real NASA POWER payloads (no network).

Real API facts this payload mimics (verified against the live API):
values live at properties.parameter.<PARAM>.<YYYYMMDD>, the fill value is
-999.0, and the most recent days arrive as -999 until NASA processes them.
"""

import numpy as np
import pandas as pd

import config
from src.data import _parse_power_response

DATES = [
    "20240101",
    "20240102",
    "20240103",
    "20240104",
    "20240105",  # target = -999 -> row dropped
    "20240106",
    "20240107",  # trailing: all params -999 -> dropped
    "20240108",  # trailing
    "20240109",  # trailing
]
TRAILING = {"20240107", "20240108", "20240109"}


def _payload() -> dict:
    """Handcrafted payload matching the real API's JSON shape, built from
    config.NASA_POWER_PARAMETERS so it fails loudly on schema drift."""
    parameter = {}
    for i, p in enumerate(config.NASA_POWER_PARAMETERS):
        vals = {d: float(i + 1) + j * 0.1 for j, d in enumerate(DATES)}
        for d in TRAILING:
            vals[d] = -999.0
        parameter[p] = vals

    # mid-series gaps: one in the target (drops the row), one in a
    # non-target column (row kept, value becomes NaN)
    parameter[config.TARGET_COL]["20240105"] = -999.0
    parameter["T2M"]["20240103"] = -999.0

    # the API returns date keys in order; feed them scrambled to pin the sort
    parameter = {
        p: dict(sorted(vals.items(), reverse=True)) for p, vals in parameter.items()
    }
    return {"properties": {"parameter": parameter}}


def test_parse_power_response():
    df = _parse_power_response(_payload())

    # sorted DatetimeIndex despite scrambled input keys
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

    # trailing not-yet-processed days and the target-missing day are dropped
    expected_dates = pd.to_datetime(["20240101", "20240102", "20240103", "20240104", "20240106"])
    assert df.index.equals(pd.DatetimeIndex(expected_dates))

    # no -999 survives anywhere; non-target gaps become NaN, row retained
    assert not (df == -999.0).any().any()
    assert np.isnan(df.loc["2024-01-03", "T2M"])
    assert not df[config.TARGET_COL].isna().any()

    assert set(df.columns) == set(config.NASA_POWER_PARAMETERS)
