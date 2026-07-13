"""_parse_power_response contract: the -999 / calendar-gap handling that
fetch_history applies to real NASA POWER payloads (no network).

Real API facts this payload mimics (verified against the live API):
values live at properties.parameter.<PARAM>.<YYYYMMDD>, the fill value is
-999.0, and the most recent days arrive as -999 until NASA processes them.

The parser keeps every calendar day between the first and last date — gaps
become explicit all-NaN rows, missing values become NaN, and nothing is
dropped here. Target-less rows are removed later, by the dropna at the end
of feature engineering, AFTER shift/rolling have seen the true calendar.
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
    "20240105",  # target = -999 -> target NaN, row kept
    "20240106",
    "20240107",  # trailing: all params -999 -> all-NaN row, kept
    "20240108",  # trailing
    "20240109",  # trailing
]
TRAILING = {"20240107", "20240108", "20240109"}


def _payload(drop_dates: tuple[str, ...] = ()) -> dict:
    """Handcrafted payload matching the real API's JSON shape, built from
    config.NASA_POWER_PARAMETERS so it fails loudly on schema drift.
    `drop_dates` deletes date keys entirely (a true calendar gap)."""
    parameter = {}
    for i, p in enumerate(config.NASA_POWER_PARAMETERS):
        vals = {d: float(i + 1) + j * 0.1 for j, d in enumerate(DATES)}
        for d in TRAILING:
            vals[d] = -999.0
        for d in drop_dates:
            del vals[d]
        parameter[p] = vals

    # mid-series -999s: one in the target, one in a non-target column
    parameter[config.TARGET_COL]["20240105"] = -999.0
    parameter["T2M"]["20240103"] = -999.0

    # the API returns date keys in order; feed them scrambled to pin the sort
    parameter = {
        p: dict(sorted(vals.items(), reverse=True)) for p, vals in parameter.items()
    }
    return {"properties": {"parameter": parameter}}


def test_parse_power_response():
    df = _parse_power_response(_payload())

    # continuous daily DatetimeIndex despite scrambled input keys
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.equals(pd.date_range("2024-01-01", "2024-01-09", freq="D"))

    # no -999 survives anywhere; gaps become NaN, all rows retained
    assert not (df == -999.0).any().any()
    assert np.isnan(df.loc["2024-01-03", "T2M"])

    # target is NaN exactly on the mid-series -999 day + the trailing days
    nan_days = set(df.index[df[config.TARGET_COL].isna()].strftime("%Y%m%d"))
    assert nan_days == {"20240105"} | TRAILING

    assert set(df.columns) == set(config.NASA_POWER_PARAMETERS)


def test_parse_power_response_restores_deleted_calendar_days():
    """A date key entirely absent from the payload must come back as an
    explicit all-NaN row — the index stays gapless so downstream
    shift/rolling operations are calendar-true."""
    df = _parse_power_response(_payload(drop_dates=("20240104",)))

    assert df.index.equals(pd.date_range("2024-01-01", "2024-01-09", freq="D"))
    assert df.loc["2024-01-04"].isna().all()
