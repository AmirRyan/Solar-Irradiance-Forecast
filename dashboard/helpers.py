"""Pure dashboard helpers — no dash imports, testable without a server.

Everything here is presentation-side: input resolution/validation and a
small in-process result memo. Modeling logic stays in src/ (pipeline.py is
the single code path both callers share).
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict

import config


def validate_latlon(lat, lon):
    """Return (lat, lon) as floats, or a human-readable error string."""
    if lat is None or lon is None:
        return "Enter both latitude and longitude."
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return "Latitude and longitude must be numbers."
    if not -90 <= lat <= 90:
        return "Latitude must be between -90 and 90."
    if not -180 <= lon <= 180:
        return "Longitude must be between -180 and 180."
    return lat, lon


def resolve_location(trigger, preset, click_data, lat, lon):
    """Map the input that FIRED to a location — the trigger wins, so stale
    values in other inputs can never shadow the user's latest action.

    Returns (lat, lon), an error string (invalid manual entry), or None
    (trigger carried nothing usable — e.g. the preset was cleared).
    """
    if trigger == "go":
        return validate_latlon(lat, lon)
    if trigger == "map":
        if click_data and click_data.get("points"):
            p = click_data["points"][0]
            if "lat" in p and "lon" in p:
                return float(p["lat"]), float(p["lon"])
        return None
    if trigger == "preset":
        loc = config.DEMO_LOCATIONS.get(preset)
        if loc:
            return loc["lat"], loc["lon"]
        return None
    return None


def memo_key(lat: float, lon: float) -> tuple[float, float]:
    """Round to 2dp — deliberately the same key as data.py's parquet cache
    (_cache_path), so the two cache layers agree on location identity."""
    return (round(lat, 2), round(lon, 2))


def preset_name_for(lat: float, lon: float) -> str | None:
    """The demo preset at this location (same 2dp identity as the caches), or
    None. Only presets have LSTM forecasts."""
    key = memo_key(lat, lon)
    for name, loc in config.DEMO_LOCATIONS.items():
        if memo_key(loc["lat"], loc["lon"]) == key:
            return name
    return None


def align_seq_forecasts(saved, test_index):
    """Saved LSTM forecasts reindexed to the current test window, or None if
    they share no scored day with it. Caches refresh every ~10 days, so the
    window can shift after train.py ran; only overlapping days are shown."""
    aligned = saved.reindex(test_index)
    return aligned if aligned["p50"].notna().any() else None


class ResultMemo:
    """Thread-safe LRU memo of pipeline results, so revisiting a location in
    a live demo skips the refit. Presentation-side latency management only —
    pipeline.py stays pure and refits on every call."""

    def __init__(self, maxsize: int = 20):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, lat: float, lon: float):
        key = memo_key(lat, lon)
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, lat: float, lon: float, value) -> None:
        key = memo_key(lat, lon)
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Plain-English presentation helpers
# ---------------------------------------------------------------------------
_WINDOW_WORDS = {1: "yesterday", 2: "2 days ago", 3: "3 days ago", 7: "a week ago",
                 14: "2 weeks ago"}


def plain_feature_name(col: str) -> str:
    """Readable label for an engineered feature column (the raw name is still
    shown on hover). Falls back to the column name for anything unknown, so a
    new feature never breaks the chart — it just stays technical."""
    target = config.TARGET_COL
    if m := re.fullmatch(rf"{target}_lag(\d+)", col):
        return f"Sunlight {_WINDOW_WORDS.get(int(m[1]), f'{m[1]} days ago')}"
    if m := re.fullmatch(rf"{target}_roll(\d+)_mean", col):
        return f"Average sunlight, last {m[1]} days"
    if m := re.fullmatch(rf"{target}_roll(\d+)_std", col):
        return f"Day-to-day swings in sunlight, last {m[1]} days"
    fixed = {
        "clearness_index_lag1": "Clear vs cloudy sky yesterday",
        "T2M_lag1": "Temperature yesterday",
        "RH2M_lag1": "Humidity yesterday",
        "WS2M_lag1": "Wind speed yesterday",
        "PRECTOTCORR_lag1": "Rain yesterday",
        "doy_sin": "Time of year (1 of 2)",
        "doy_cos": "Time of year (2 of 2)",
    }
    return fixed.get(col, col)


def summary_text(place: str, mae: float, persistence_mae: float, coverage: float,
                 mean_irradiance: float, test_days: int) -> list[str]:
    """Three plain sentences summarising a result, for the headline card."""
    pct = mae / mean_irradiance if mean_irradiance > 0 else float("nan")
    better = (persistence_mae - mae) / persistence_mae if persistence_mae > 0 else 0.0
    return [
        f"For {place}, the model was tested on {test_days} past days it had never "
        f"seen. On average its guess was off by {mae:.2f} kWh/m² per day, "
        f"about {pct:.0%} of the average daily sunlight ({mean_irradiance:.1f}).",
        f"That is {abs(better):.0%} {'smaller' if better >= 0 else 'larger'} than "
        f"simply guessing \u201ctomorrow equals today\u201d, which was off by "
        f"{persistence_mae:.2f}.",
        f"The shaded range is meant to contain the real value 8 days out of 10; "
        f"it actually did on {coverage:.0%} of days.",
    ]
