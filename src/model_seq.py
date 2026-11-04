"""Phase-4 comparison model: single-layer LSTM + attention, quantile outputs.

Classical-vs-deep-learning comparison against QuantileGBM; it is NOT required
to beat LightGBM. Runs on the pre-cached demo locations via train.py only
(training per request is too slow for interactive latency), so the dashboard
reads its saved forecasts instead of importing this module.

API matches QuantileGBM: fit(train) / predict(df) -> p10/p50/p90 columns.

Input is a sliding window of the last `window` rows of config.FEATURES. Those
features are already lagged, so the window ending at day t never contains day
t's target. A window that spans a calendar gap (engineer_features drops the
days around a gap) is not a real 30-day history, so its prediction is NaN
rather than a misaligned guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

import config

# LightGBM and torch each bundle an OpenMP runtime. In one process, a
# multithreaded torch op after a LightGBM fit deadlocks on macOS (0% CPU,
# measured), and importing torch before lightgbm segfaults. The net is tiny
# (hidden 32), so a single torch thread costs nothing. Keep lightgbm imported
# first (pipeline.py does) and this at 1.
torch.set_num_threads(1)


class _AttentionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float, n_out: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                                  # (B, T, H)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (B, T)
        context = (w.unsqueeze(-1) * h).sum(dim=1)           # (B, H)
        return self.head(self.drop(context))


def _pinball(pred: torch.Tensor, y: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred.dtype).unsqueeze(0)
    diff = y.unsqueeze(1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


class SeqModel:
    def __init__(self, quantiles: list[float] | None = None, **overrides):
        self.quantiles = quantiles or config.QUANTILES
        self.params = {**config.SEQ_PARAMS, **overrides}
        self.net: _AttentionLSTM | None = None
        self._tail: pd.DataFrame | None = None

    # -- windows ------------------------------------------------------------
    def _windows(self, df: pd.DataFrame):
        """(windows, end_index): one window per row that has `window`
        consecutive calendar days ending at it."""
        w = self.params["window"]
        x = ((df[config.FEATURES].to_numpy(dtype=np.float64) - self._mu) / self._sd)
        if len(df) < w:
            return np.empty((0, w, x.shape[1]), dtype=np.float32), df.index[:0]
        win = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)  # (N-w+1, F, w)
        win = win.transpose(0, 2, 1).astype(np.float32)
        end = df.index[w - 1:]
        start = df.index[: len(df) - w + 1]
        contiguous = (end - start) == pd.Timedelta(days=w - 1)
        return win[contiguous], end[contiguous]

    # -- fit ----------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "SeqModel":
        p = self.params
        torch.set_num_threads(1)  # per calling thread: the dashboard fits in a worker
        torch.manual_seed(p["seed"])
        feats = train[config.FEATURES].to_numpy(dtype=np.float64)
        self._mu = feats.mean(axis=0)
        self._sd = np.maximum(feats.std(axis=0), 1e-6)
        y = train[config.TARGET_COL]
        self._y_mu, self._y_sd = float(y.mean()), float(max(y.std(), 1e-6))

        X, end = self._windows(train)
        if len(X) < 50:
            raise ValueError(f"Only {len(X)} full {p['window']}-day windows to fit on.")
        y_s = ((y.loc[end].to_numpy() - self._y_mu) / self._y_sd).astype(np.float32)

        n_val = max(int(len(X) * p["val_fraction"]), 1)  # chronological tail
        Xtr, ytr = torch.from_numpy(X[:-n_val]), torch.from_numpy(y_s[:-n_val])
        Xva, yva = torch.from_numpy(X[-n_val:]), torch.from_numpy(y_s[-n_val:])

        self.net = _AttentionLSTM(len(config.FEATURES), p["hidden"], p["dropout"],
                                  len(self.quantiles))
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"],
                               weight_decay=p["weight_decay"])
        best, best_state, stale = float("inf"), None, 0
        for _ in range(p["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), p["batch_size"]):
                idx = perm[i:i + p["batch_size"]]
                opt.zero_grad()
                _pinball(self.net(Xtr[idx]), ytr[idx], self.quantiles).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(_pinball(self.net(Xva), yva, self.quantiles))
            if val < best - 1e-5:
                best, stale = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self._tail = train.iloc[-(p["window"] - 1):]
        return self

    # -- predict ------------------------------------------------------------
    def predict(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        """Columns p10/p50/p90 indexed like `df`; NaN where the row has no full
        contiguous window. `context` supplies the history before `df` (default:
        the tail of the fit data, used only when `df` starts after it)."""
        torch.set_num_threads(1)
        if context is None and self._tail is not None \
                and len(df) and df.index.min() > self._tail.index.max():
            context = self._tail
        full = df if context is None else pd.concat([context, df])
        full = full[~full.index.duplicated(keep="last")].sort_index()

        X, end = self._windows(full)
        cols = [f"p{int(q * 100)}" for q in self.quantiles]
        out = pd.DataFrame(np.nan, index=df.index, columns=cols)
        keep = end.isin(df.index)
        if keep.any():
            with torch.no_grad():
                raw = self.net(torch.from_numpy(X[keep])).numpy()
            preds = np.sort(raw * self._y_sd + self._y_mu, axis=1)  # no crossing
            out.loc[end[keep], cols] = preds
        return out
