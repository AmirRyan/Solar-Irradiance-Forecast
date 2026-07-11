"""RegimeClusterer contract (no network). Guards the naming heuristic that
was rewritten when CLOUD_AMT was dropped (Overcast = lowest clearness)."""

import config
from src.cluster import RegimeClusterer
from src.data import engineer_features


def test_regime_clusterer_names_and_populates_all_regimes(synthetic_raw):
    df = engineer_features(synthetic_raw)

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
    # flatten the seasonal cycle: constant clear-sky, target = clrsky * cloud
    flat["CLRSKY_SFC_SW_DWN"] = 6.5
    cloudiness = (flat["ALLSKY_SFC_SW_DWN"] / synthetic_raw["CLRSKY_SFC_SW_DWN"]).clip(0.05, 1)
    flat["ALLSKY_SFC_SW_DWN"] = 6.5 * cloudiness
    df = engineer_features(flat)

    clusterer = RegimeClusterer().fit(df)
    labels = set(clusterer.label_map.values())

    assert "Winter Low" not in labels
    assert config.REGIME_LOW_SEASONALITY_NAME in labels
    expected = (set(config.REGIME_NAMES) - {"Winter Low"}) | {
        config.REGIME_LOW_SEASONALITY_NAME
    }
    assert labels == expected
