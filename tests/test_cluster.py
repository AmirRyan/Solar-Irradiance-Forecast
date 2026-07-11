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
