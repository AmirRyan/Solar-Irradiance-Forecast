"""Weather regime clustering (KMeans) + per-regime error metrics.

An ANALYSIS layer — regimes are used for the per-regime error breakdown and
dashboard coloring, never as a modeling input.

OPEN QUESTION (see CLAUDE.md): fit once on demo-location data and .predict()
on new locations, vs. refit per location. Refitting on a single location's
limited history risks unstable clusters. Current scaffold refits per location
(simplest); revisit before finalizing the error-analysis writeup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config

# Daily weather descriptors used to define regimes (all same-day values —
# fine here because regimes describe the day being evaluated, they are not
# predictive features).
REGIME_INPUT_COLS = ["clearness_index_lag1", "CLOUD_AMT_lag1", "T2M_lag1", "doy_cos"]


class RegimeClusterer:
    def __init__(self, n_regimes: int = config.N_REGIMES, seed: int = 0):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed)
        self.label_map: dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> "RegimeClusterer":
        X = self.scaler.fit_transform(df[REGIME_INPUT_COLS])
        self.kmeans.fit(X)
        self._name_clusters(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.scaler.transform(df[REGIME_INPUT_COLS])
        ids = self.kmeans.predict(X)
        names = [self.label_map[i] for i in ids]
        return pd.Series(names, index=df.index, name="regime")

    def _name_clusters(self, df: pd.DataFrame) -> None:
        """Map anonymous cluster ids to human regime names using cluster
        characteristics: high clearness -> Clear Sky, high cloud -> Overcast,
        low irradiance + winter -> Winter Low, remainder -> Transitional.
        Heuristic — inspect and adjust once real data is flowing."""
        X = self.scaler.transform(df[REGIME_INPUT_COLS])
        ids = self.kmeans.predict(X)
        stats = df[REGIME_INPUT_COLS].groupby(ids).mean()

        remaining = set(stats.index)
        self.label_map = {}

        clear = stats.loc[list(remaining), "clearness_index_lag1"].idxmax()
        self.label_map[clear] = "Clear Sky"
        remaining.discard(clear)

        overcast = stats.loc[list(remaining), "CLOUD_AMT_lag1"].idxmax()
        self.label_map[overcast] = "Overcast"
        remaining.discard(overcast)

        winter = stats.loc[list(remaining), "doy_cos"].idxmax()  # doy_cos peaks midwinter (Jan)
        self.label_map[winter] = "Winter Low"
        remaining.discard(winter)

        for i in remaining:
            self.label_map[i] = "Transitional"


def per_regime_rmse(
    y_true: pd.Series, y_pred: pd.Series, regimes: pd.Series
) -> pd.DataFrame:
    """The primary error-analysis lens: RMSE (and count) per weather regime."""
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "regime": regimes}).dropna()
    rows = []
    for name, g in df.groupby("regime"):
        rows.append(
            {
                "regime": name,
                "n_days": len(g),
                "rmse": float(np.sqrt(np.mean((g["y"] - g["yhat"]) ** 2))),
                "mae": float(np.mean(np.abs(g["y"] - g["yhat"]))),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
