"""Pure dashboard helpers — no dash imports, testable without a server.

Everything here is presentation-side: input resolution/validation and a
small in-process result memo. Modeling logic stays in src/ (pipeline.py is
the single code path both callers share).
"""

from __future__ import annotations

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
