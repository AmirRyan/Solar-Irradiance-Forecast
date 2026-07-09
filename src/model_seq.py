"""Week-4 stretch: attention LSTM / TFT-lite comparison model.

STATUS: stub. Do not build until the week-4 go/no-go decision (see CLAUDE.md).

Framing: classical-vs-deep-learning comparison. It is NOT required to beat
LightGBM to be a useful result.

OPEN QUESTION (resolve before wiring into dashboard/app.py): training an
LSTM/TFT-lite per-request is not feasible at interactive latency, so this
almost certainly runs only against the pre-cached demo locations via
train.py — confirm before adding it to the live app.

Planned shape (kept API-compatible with QuantileGBM so the dashboard can
render it as a second forecast line with zero special-casing):

    class SeqModel:
        def fit(self, train: pd.DataFrame) -> "SeqModel": ...
        def predict(self, df: pd.DataFrame) -> pd.DataFrame:  # p10/p50/p90
            ...

Implementation notes for when this gets built:
- Input: sliding windows over the same engineered features (config.FEATURES),
  standardized; window length ~30 days.
- Quantile (pinball) loss heads for P10/P50/P90 to stay comparable with the
  LightGBM outputs and the pinball-loss evaluation.
- Keep it compact (single LSTM layer + attention, or a TFT-lite) — the point
  is the comparison, not the architecture.
"""

from __future__ import annotations


class SeqModel:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "model_seq.py is a week-4 stretch stub — see module docstring "
            "and the open design questions in CLAUDE.md before implementing."
        )
