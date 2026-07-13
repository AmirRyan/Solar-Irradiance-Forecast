"""RegimeClusterer contract (no network). Guards the naming heuristic that
was rewritten when CLOUD_AMT was dropped (Overcast = lowest clearness)."""

import numpy as np

import config
from src.cluster import RegimeClusterer
from src.data import engineer_features


def test_regime_clusterer_names_and_populates_all_regimes(synthetic_raw, synthetic_lat):
    df = engineer_features(synthetic_raw, synthetic_lat)

    clusterer = RegimeClusterer().fit(df)
    # predict on the fit frame: every KMeans cluster has at least one
    # training point, so all four regimes must appear
    regimes = clusterer.predict(df)

    assert set(regimes) <= set(config.REGIME_NAMES)
    assert set(clusterer.label_map.values()) == set(config.REGIME_NAMES)

    counts = regimes.value_counts()
    assert set(counts.index) == set(config.REGIME_NAMES)
    assert (counts > 0).all()


def test_equatorial_location_gets_no_winter_label(synthetic_raw):
    """Locations with no seasonal irradiance cycle (equatorial) must get the
    neutral low-seasonality label instead of a meaningless "Winter Low"."""
    flat = synthetic_raw.copy()
    # flatten the seasonal cycle: a purely cloud-driven target with no annual
    # trend, engineered at the equator (lat=0), where TOA is ~constant too
    rng = np.random.default_rng(7)
    flat[config.TARGET_COL] = 6.5 * rng.uniform(0.3, 1.0, len(flat))
    df = engineer_features(flat, lat=0.0)

    clusterer = RegimeClusterer().fit(df)
    labels = set(clusterer.label_map.values())

    assert "Winter Low" not in labels
    assert config.REGIME_LOW_SEASONALITY_NAME in labels
    expected = (set(config.REGIME_NAMES) - {"Winter Low"}) | {
        config.REGIME_LOW_SEASONALITY_NAME
    }
    assert labels == expected
