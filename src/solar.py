"""Analytic solar geometry. Pure numpy — no network, no config, no project
imports, so it is usable from anywhere (tests, notebooks, data.py).

toa_irradiance() is the clearness-index denominator. It replaces NASA
POWER's CLRSKY_SFC_SW_DWN parameter, which turned out to be CERES-derived
and to lag ~3 months behind the near-real-time parameters (the same failure
mode as the already-removed CLOUD_AMT) — every missing day was killing the
following day's engineered row via clearness_index_lag1.
"""

from __future__ import annotations

import numpy as np

GSC = 1361.0  # solar constant, W/m^2


def toa_irradiance(lat_deg: float, doy: np.ndarray | int) -> np.ndarray | float:
    """Daily extraterrestrial (top-of-atmosphere) irradiance on a horizontal
    surface, in kWh/m^2/day. Standard daily formula (Duffie & Beckman):

      H0 = (24*3600/pi) * Gsc * E0 * (cos(phi)cos(delta)sin(ws) + ws sin(phi)sin(delta))

    with Gsc = 1361 W/m^2, E0 the eccentricity correction factor, delta the
    solar declination, ws the sunset hour angle. Vectorized over doy.

    The arccos argument is clamped to [-1, 1] so polar latitudes don't NaN:
    polar night -> ws = 0 -> H0 = 0 exactly; polar day -> ws = pi.
    """
    doy = np.asarray(doy, dtype=float)
    phi = np.radians(lat_deg)

    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))

    h0 = (24.0 * 3600.0 / np.pi) * GSC * e0 * (
        np.cos(phi) * np.cos(delta) * np.sin(ws) + ws * np.sin(phi) * np.sin(delta)
    )
    h0 = np.maximum(h0, 0.0) / 3.6e6  # J/m^2/day -> kWh/m^2/day
    return float(h0) if doy.ndim == 0 else h0
