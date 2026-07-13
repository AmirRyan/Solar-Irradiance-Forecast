"""toa_irradiance contract: pure solar geometry, no config/network/fixtures.

Reference values are cross-checked against the Duffie & Beckman worked
example (43N, Apr 15: 33.8 MJ/m^2 with their Gsc=1367; ours reads ~0.4%
lower with the modern Gsc=1361) and the FAO-56 extraterrestrial-radiation
tables. NOTE: the task brief quoted ~3.6 kWh/m^2/day for 45N at winter
solstice, but the standard formula (and FAO-56) give ~2.9 there — 3.6-3.7
is the 40N value. Both latitudes are pinned below, against the physically
correct numbers.
"""

import numpy as np
import pytest

from src.solar import toa_irradiance


@pytest.mark.parametrize(
    "lat, doy, expected",
    [
        (0.0, 80, 10.3),    # equator, equinox
        (45.0, 172, 11.6),  # 45N, summer solstice
        (45.0, 355, 2.89),  # 45N, winter solstice (FAO-56: ~10.4 MJ/m^2/day)
        (40.0, 355, 3.74),  # 40N, winter solstice (the brief's "~3.6" anchor)
        (43.0, 105, 9.34),  # D&B example 1.10.1 (33.8 MJ/m^2 at Gsc=1367)
    ],
)
def test_reference_values(lat, doy, expected):
    assert toa_irradiance(lat, doy) == pytest.approx(expected, rel=0.05)


def test_nonnegative_and_finite_everywhere():
    doy = np.arange(1, 366)
    for lat in range(-90, 91, 5):
        h0 = toa_irradiance(float(lat), doy)
        assert np.isfinite(h0).all(), f"non-finite H0 at lat={lat}"
        assert (h0 >= 0).all(), f"negative H0 at lat={lat}"


def test_polar_night_is_exactly_zero():
    assert toa_irradiance(70.0, 355) == 0.0


def test_no_nans_below_polar_circle():
    doy = np.arange(1, 366)
    for lat in (-66.0, -45.0, 0.0, 45.0, 66.0):
        assert not np.isnan(toa_irradiance(lat, doy)).any()


def test_hemispheric_symmetry():
    # 45S midwinter (doy 172) vs 45N midwinter (doy 355). Not exact: the
    # eccentricity correction (perihelion in early January) makes the two
    # differ by ~6%, so the tolerance is 7%, not 5%.
    assert toa_irradiance(-45.0, 172) == pytest.approx(
        toa_irradiance(45.0, 355), rel=0.07
    )


def test_scalar_in_scalar_out_vector_in_vector_out():
    assert isinstance(toa_irradiance(45.0, 172), float)
    out = toa_irradiance(45.0, np.array([80, 172, 355]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)
